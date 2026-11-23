import os, json, warnings, torch, numpy as np, evaluate
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
from model.nltoolkit import init_nltk, postprocess_text
from utils.converter import batch_text2traj
from utils.dataloader import get_dataloader
from utils.homography import generate_homography, image2world
from utils.postprocessor import postprocess_trajectory

warnings.filterwarnings("ignore")
logger = get_logger(__name__)
if torch.cuda.is_available(): torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def is_valid_model_dir(model_path):
    if not os.path.isdir(model_path): return False
    if not os.path.isfile(os.path.join(model_path, "config.json")): return False
    weight_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") or f.endswith(".bin")]
    return len(weight_files) > 0

def resolve_test_model_path(checkpoint_root, test_model_name):
    best_model_path = os.path.join(checkpoint_root, "best_model"); last_model_path = os.path.join(checkpoint_root, "last_model")
    if test_model_name == "best_model":
        if is_valid_model_dir(best_model_path): return best_model_path, "best_model"
        raise FileNotFoundError(f"\nBest model was requested, but a complete best_model was not found.\nChecked:\n{best_model_path}")
    elif test_model_name == "last_model":
        if is_valid_model_dir(last_model_path): return last_model_path, "last_model"
        if is_valid_model_dir(checkpoint_root): print("\n" + "=" * 70); print("NOTE"); print("last_model/ does not contain a complete model."); print("The final model is stored directly in the checkpoint root."); print("Using:", checkpoint_root); print("=" * 70); return checkpoint_root, "last_model(root)"
        raise FileNotFoundError(f"\nCannot find a complete last model.\nChecked:\n1. {last_model_path}\n2. {checkpoint_root}")
    elif test_model_name == "root":
        if is_valid_model_dir(checkpoint_root): return checkpoint_root, "root"
        raise FileNotFoundError(f"No complete model found in:\n{checkpoint_root}")
    else: raise ValueError("test_model must be one of: 'best_model', 'last_model', 'root'")

@torch.no_grad()
def test(cfg):
    init_nltk(); accelerator_log_kwargs = {}; checkpoint_root = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name)
    if cfg.use_logger: accelerator_log_kwargs["log_with"] = cfg.logger_type; accelerator_log_kwargs["project_dir"] = checkpoint_root
    accelerator = Accelerator(mixed_precision="bf16", **accelerator_log_kwargs)
    if cfg.seed is not None: set_seed(cfg.seed)

    test_model_name = getattr(cfg, "test_model", "best_model"); model_path, actual_model_name = resolve_test_model_path(checkpoint_root, test_model_name)
    if accelerator.is_main_process: print("\n" + "=" * 70); print("TEST MODEL"); print("=" * 70); print("Checkpoint root :", checkpoint_root); print("Requested model :", test_model_name); print("Actual model    :", actual_model_name); print("Model path      :", model_path); print("=" * 70)

    mo_path = os.path.join("./datasets", cfg.dataset_name, "Mo"); lng_path = os.path.join("./datasets", cfg.dataset_name, "Lng")
    print("\nOriginal trajectory data:"); print("Mo :", mo_path); print("Lng:", lng_path)
    dataloader = get_dataloader(mo_path, "test", cfg.obs_len, cfg.pred_len, batch_size=100000000)
    obs_traj = dataloader.dataset.obs_traj.numpy(); pred_traj = dataloader.dataset.pred_traj.numpy(); non_linear_ped = dataloader.dataset.non_linear_ped.numpy(); homography = dataloader.dataset.homography; scene_id = dataloader.dataset.scene_id; scene_img = dataloader.dataset.scene_img; scene_map = dataloader.dataset.scene_map; seq_start_end = dataloader.dataset.seq_start_end

    batch_size_per_gpu = obs_traj.shape[0] // accelerator.state.num_processes + 1
    if batch_size_per_gpu < cfg.per_device_inference_batch_size: print(f"per_device_inference_batch_size is automatically reduced from {cfg.per_device_inference_batch_size} to {batch_size_per_gpu}."); cfg.per_device_inference_batch_size = batch_size_per_gpu

    for k, v in homography.items(): cfg.image_scale_down = 0.25; homography[k] = v.copy() @ generate_homography(scale=cfg.image_scale_down)

    preprocessed_test_dataset_name = f"{cfg.dataset_name}-test-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json"; preprocessed_dataset_path = os.path.join(cfg.dataset_path, "preprocessed"); test_json_path = os.path.join(preprocessed_dataset_path, preprocessed_test_dataset_name)
    if not os.path.exists(test_json_path): raise FileNotFoundError(f"Test dataset not found:\n{test_json_path}")
    print("\nTest dataset:"); print(test_json_path)
    raw_datasets = load_dataset(test_json_path.split(".")[-1], data_files={"test": test_json_path}, cache_dir=cfg.cache_dir)

    print("\nLoading LOCAL Qwen model..."); config = AutoConfig.from_pretrained(model_path, trust_remote_code=False, cache_dir=cfg.cache_dir, local_files_only=True)
    print("\nLoading tokenizer from checkpoint:"); print(model_path)
    try: tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    except Exception as e: print("\nTokenizer not found in model checkpoint."); print("Fallback tokenizer:", cfg.tokenizer_name); print("Reason:", e); tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True, local_files_only=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None: raise ValueError("Tokenizer has neither PAD nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=False, cache_dir=cfg.cache_dir, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None: model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None: model.generation_config.eos_token_id = tokenizer.eos_token_id

    test_text = "Vessl 0 has no collision risk."; test_enc = tokenizer(test_text, add_special_tokens=False)
    print("\n" + "=" * 70); print("TOKENIZER / MODEL CHECK"); print("=" * 70); print("Tokenizer class :", tokenizer.__class__.__name__); print("Tokenizer vocab :", len(tokenizer)); print("Model vocab     :", model.get_input_embeddings().num_embeddings); print("Config vocab    :", model.config.vocab_size); print("PAD             :", tokenizer.pad_token_id); print("BOS             :", tokenizer.bos_token_id); print("EOS             :", tokenizer.eos_token_id); print("UNK             :", tokenizer.unk_token_id); print("Test text       :", test_text); print("Test IDs        :", test_enc["input_ids"]); print("Test Tokens     :", tokenizer.convert_ids_to_tokens(test_enc["input_ids"])); print("Test Decode     :", tokenizer.decode(test_enc["input_ids"], skip_special_tokens=False)); print("=" * 70)
    assert len(tokenizer) > 1000, f"Abnormal tokenizer vocab: {len(tokenizer)}"
    if tokenizer.unk_token_id is not None: assert tokenizer.unk_token_id not in test_enc["input_ids"], "Tokenizer still produces <unk>."
    assert model.get_input_embeddings().num_embeddings == len(tokenizer), f"Model vocab {model.get_input_embeddings().num_embeddings} != tokenizer vocab {len(tokenizer)}"

    if accelerator.is_main_process:
        def count_parameters(model): return sum(p.numel() for p in model.parameters())
        print("\n" + "=" * 70); print("MODEL INFORMATION"); print("=" * 70); print("Model class     :", model.__class__.__name__); print("Config class    :", config.__class__.__name__); print("Tokenizer class :", tokenizer.__class__.__name__); print("Vocabulary size :", len(tokenizer)); print("Parameters      :", count_parameters(model)); print("Parameters (M)  :", count_parameters(model) / 1e6); print("Model dtype     :", next(model.parameters()).dtype); print("PAD token       :", tokenizer.pad_token_id); print("BOS token       :", tokenizer.bos_token_id); print("EOS token       :", tokenizer.eos_token_id); print("UNK token       :", tokenizer.unk_token_id); print("=" * 70)

    column_names = raw_datasets["test"].column_names; history_column = cfg.history_column; future_column = cfg.future_column
    if history_column not in column_names: raise ValueError(f"history_column '{history_column}' not found.")
    if future_column not in column_names: raise ValueError(f"future_column '{future_column}' not found.")

    def encode_source(text): return tokenizer(text, max_length=cfg.max_source_length, truncation=True, add_special_tokens=False)["input_ids"]

    def encode_target(text):
        max_target_length = cfg.max_target_length - 1 if tokenizer.eos_token_id is not None else cfg.max_target_length; ids = tokenizer(text, max_length=max_target_length, truncation=True, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None: ids = ids + [tokenizer.eos_token_id]
        return ids

    def preprocess_function(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for source, target in zip(examples[history_column], examples[future_column]):
            source_ids = encode_source(source); target_ids = encode_target(target); model_inputs["input_ids"].append(source_ids); model_inputs["attention_mask"].append([1] * len(source_ids)); model_inputs["labels"].append(target_ids)
        return model_inputs

    test_dataset = raw_datasets["test"].map(preprocess_function, batched=True, remove_columns=column_names, load_from_cache_file=False, desc="Tokenizing test dataset")

    if len(test_dataset) > 0:
        sample = test_dataset[0]; sample_ids = sample["input_ids"][:50]; valid_labels = [x for x in sample["labels"] if x != -100]
        print("\n" + "=" * 70); print("TOKENIZED TEST SAMPLE"); print("=" * 70); print("Input IDs     :", sample_ids); print("Input Tokens  :", tokenizer.convert_ids_to_tokens(sample_ids)); print("Decoded Input :", tokenizer.decode(sample["input_ids"], skip_special_tokens=True)[:500]); print("Decoded Target:", tokenizer.decode(valid_labels, skip_special_tokens=True)); print("=" * 70)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, label_pad_token_id=-100, pad_to_multiple_of=8)
    eval_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=cfg.per_device_inference_batch_size, pin_memory=True)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader); model.eval()

    if accelerator.is_main_process:
        print("\n" + "=" * 70); print("DEVICE / PRECISION CHECK"); print("=" * 70); print("Accelerator device :", accelerator.device); print("Model device       :", next(model.parameters()).device); print("Model dtype        :", next(model.parameters()).dtype); print("Mixed precision    :", accelerator.mixed_precision)
        if torch.cuda.is_available(): print("GPU                :", torch.cuda.get_device_name(0)); print(f"GPU allocated GB   : {torch.cuda.memory_allocated(0) / 1024**3:.3f}"); print(f"GPU reserved GB    : {torch.cuda.memory_reserved(0) / 1024**3:.3f}")
        print("=" * 70)

    rouge_metric = evaluate.load("rouge"); all_obs = np.array(raw_datasets["test"]["obs_traj"]).astype(np.float32); all_gts = np.array(raw_datasets["test"]["pred_traj"]).astype(np.float32); all_preds = []; error_ids = []
    progress_bar = tqdm(total=len(obs_traj), desc="Generating", disable=not accelerator.is_local_main_process); processed_samples = 0

    for step, batch in enumerate(eval_dataloader):
        input_length = batch["input_ids"].shape[1]

        if cfg.deterministic:
            generated_tokens = accelerator.unwrap_model(model).generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=cfg.max_target_length, num_beams=cfg.num_beams, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True); current_num_samples = 1
        else:
            generated_tokens = accelerator.unwrap_model(model).generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=cfg.max_target_length, do_sample=True, num_return_sequences=cfg.num_samples, temperature=cfg.temperature, top_k=cfg.top_k, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True); current_num_samples = cfg.num_samples

        generated_tokens = generated_tokens[:, input_length:]
        generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)

        if cfg.deterministic: generated_grouped = generated_tokens.unsqueeze(1)
        else: generated_grouped = generated_tokens.reshape(-1, current_num_samples, generated_tokens.size(-1))

        generated_grouped = accelerator.gather_for_metrics(generated_grouped)
        labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=-100); labels = accelerator.gather_for_metrics(labels)
        generated_grouped_np = generated_grouped.detach().cpu().numpy(); labels_np = labels.detach().cpu().numpy(); current_batch_size = generated_grouped_np.shape[0]; labels_np = np.where(labels_np != -100, labels_np, tokenizer.pad_token_id)

        rouge_tokens = generated_grouped_np[:, 0, :]; decoded_rouge_preds = tokenizer.batch_decode(rouge_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False); decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True, clean_up_tokenization_spaces=False); decoded_rouge_preds, decoded_labels = postprocess_text(decoded_rouge_preds, decoded_labels); rouge_metric.add_batch(predictions=decoded_rouge_preds, references=decoded_labels)

        generated_flat = generated_grouped_np.reshape(-1, generated_grouped_np.shape[-1]); decoded_preds = tokenizer.batch_decode(generated_flat, skip_special_tokens=True, clean_up_tokenization_spaces=False); decoded_preds = [pred.strip() for pred in decoded_preds]

        if step == 0: print("\n" + "=" * 70); print("FIRST BATCH PREDICTIONS"); print("=" * 70); print("Labels:"); print(decoded_labels[:5]); print("\nPredictions:"); print(decoded_preds[:5]); print("=" * 70)

        traj_data = batch_text2traj(decoded_preds, frame=cfg.pred_len, dim=2)

        for pid in range(len(traj_data)):
            if traj_data[pid] is None:
                local_sample_id = pid // current_num_samples; ped_id = processed_samples + local_sample_id; error_ids.append(ped_id)
                if ped_id < len(all_obs): traj_data[pid] = np.tile(all_obs[ped_id, -1], (cfg.pred_len, 1))
                else: traj_data[pid] = np.zeros((cfg.pred_len, 2), dtype=np.float32)

        traj_data = np.stack(traj_data, axis=0).reshape(current_batch_size, current_num_samples, cfg.pred_len, 2); all_preds.append(traj_data); processed_samples += current_batch_size; progress_bar.update(current_batch_size)

    all_preds = np.concatenate(all_preds, axis=0).astype(np.float32); all_preds = all_preds[:len(obs_traj)]; progress_bar.n = len(obs_traj); progress_bar.refresh(); progress_bar.close()

    raw_rouge_result = rouge_metric.compute(use_stemmer=True); test_rouge = {k: round(float(v) * 100, 4) for k, v in raw_rouge_result.items()}

    if accelerator.is_main_process:
        print("\n" + "=" * 70); print("TEST RESULTS"); print("=" * 70); print("Dataset         :", cfg.dataset_name); print("Requested model :", test_model_name); print("Actual model    :", actual_model_name); print("Deterministic   :", cfg.deterministic); print("Num samples     :", 1 if cfg.deterministic else cfg.num_samples); print("Num beams       :", cfg.num_beams); print(f"ROUGE-1         : {test_rouge['rouge1']:.4f}"); print(f"ROUGE-2         : {test_rouge['rouge2']:.4f}"); print(f"ROUGE-L         : {test_rouge['rougeL']:.4f}"); print(f"ROUGE-Lsum      : {test_rouge['rougeLsum']:.4f}"); print("Prediction shape:", all_preds.shape); print("Invalid predictions:", len(error_ids)); print("=" * 70)

        test_result = {"dataset": str(cfg.dataset_name), "requested_model": str(test_model_name), "actual_model": str(actual_model_name), "model_path": str(model_path), "deterministic": bool(cfg.deterministic), "num_samples": int(1 if cfg.deterministic else cfg.num_samples), "num_beams": int(cfg.num_beams), "temperature": float(cfg.temperature), "top_k": int(cfg.top_k), "rouge1": float(test_rouge["rouge1"]), "rouge2": float(test_rouge["rouge2"]), "rougeL": float(test_rouge["rougeL"]), "rougeLsum": float(test_rouge["rougeLsum"]), "num_trajectories": int(all_preds.shape[0]), "invalid_predictions": int(len(error_ids))}
        test_result_path = os.path.join(checkpoint_root, f"test_results_{test_model_name}.json")
        with open(test_result_path, "w", encoding="utf-8") as f: json.dump(test_result, f, ensure_ascii=False, indent=4)
        print("\nTest result saved to:"); print(test_result_path)

    """
    if accelerator.is_main_process:
        all_preds_eval = postprocess_trajectory(all_preds.copy(), obs_traj, seq_start_end, scene_id, homography, scene_map, cfg)
        ADE, FDE = [], []
        for ped_id in range(all_preds_eval.shape[0]):
            if cfg.metric == "pixel":
                H = homography[scene_id[ped_id]]
                for sample_id in range(all_preds_eval.shape[1]): all_preds_eval[ped_id, sample_id] = image2world(all_preds_eval[ped_id, sample_id], H)
                all_gts[ped_id] = pred_traj[ped_id]
            error = np.linalg.norm(all_preds_eval[ped_id] - all_gts[ped_id], ord=2, axis=-1)
            ADE.append(np.mean(error, axis=-1).min()); FDE.append(error[:, -1].min())
        print("\n" + "=" * 70); print("TRAJECTORY RESULTS"); print("=" * 70); print("Dataset:", cfg.dataset_name); print("Trajectory number:", all_preds_eval.shape[0]); print("ADE:", np.mean(ADE)); print("FDE:", np.mean(FDE)); print("=" * 70)
    """

    return obs_traj, pred_traj, all_preds

import argparse
from utils.config import get_exp_config, print_arguments

parser = argparse.ArgumentParser(); parser.add_argument("--cfg", default="./config/config-meter.json", type=str, help="Config file path."); parser.add_argument("--dataset", default="ZSI", type=str, help="Dataset name."); parser.add_argument("--tag", default="LMTraj_Qwen35", type=str, help="Checkpoint name."); parser.add_argument("--test", default=True, action="store_true", help="Evaluation mode."); args = parser.parse_args([])

print("===== Arguments ====="); print_arguments(vars(args)); print("===== Configs ====="); cfg = get_exp_config(args.cfg); print_arguments(cfg); cfg.dataset_name = args.dataset; cfg.checkpoint_name = args.tag
if not hasattr(cfg, "test_model"): cfg.test_model = "last_model"

print("\n" + "=" * 70); print("TEST CONFIGURATION"); print("=" * 70); print("Dataset       :", cfg.dataset_name); print("Checkpoint    :", cfg.checkpoint_name); print("Test model    :", cfg.test_model); print("Deterministic :", cfg.deterministic); print("Num beams     :", cfg.num_beams); print("Num samples   :", cfg.num_samples); print("Temperature   :", cfg.temperature); print("Top-k         :", cfg.top_k); print("=" * 70)

obs_traj, pred_traj, all_preds = test(cfg)