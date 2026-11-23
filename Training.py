
import json, logging, math, os, random, shutil
import datasets, evaluate, numpy as np, torch, transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, get_scheduler
from model.nltoolkit import init_nltk, postprocess_text

logger = get_logger(__name__)

if torch.cuda.is_available(): torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def adapt_model_to_tokenizer(model, tokenizer):
    old_vocab_size = model.get_input_embeddings().weight.shape[0]; new_vocab_size = len(tokenizer); tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None: raise ValueError("Tokenizer has neither PAD nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    print('\n' + '=' * 70); print('Model class          :', model.__class__.__name__); print('Config class         :', model.config.__class__.__name__); print('Original vocab size  :', old_vocab_size); print('Tokenizer vocab size :', new_vocab_size); print('PAD token id         :', tokenizer.pad_token_id); print('BOS token id         :', tokenizer.bos_token_id); print('EOS token id         :', tokenizer.eos_token_id); print('UNK token id         :', tokenizer.unk_token_id); print('=' * 70)
    if old_vocab_size != new_vocab_size: print(f'Resize token embeddings: {old_vocab_size} -> {new_vocab_size}'); model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size; model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None: model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None: model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None: model.generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None: model.generation_config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, 'vocab_size'): model.vocab_size = new_vocab_size
    if hasattr(model, 'tie_weights'): model.tie_weights()
    print('Final embedding shape:', model.get_input_embeddings().weight.shape); print('Config vocab size    :', model.config.vocab_size); print('=' * 70 + '\n')
    assert model.get_input_embeddings().num_embeddings == len(tokenizer); assert model.config.vocab_size == len(tokenizer)
    return model

def check_tokenizer(tokenizer):
    text = "Vessl 0 has no collision risk."; enc = tokenizer(text, add_special_tokens=False); tokens = tokenizer.convert_ids_to_tokens(enc['input_ids']); unk_num = sum(x == tokenizer.unk_token_id for x in enc['input_ids']) if tokenizer.unk_token_id is not None else 0
    print('\n' + '=' * 70); print('TOKENIZER CHECK'); print('Class      :', tokenizer.__class__.__name__); print('Vocab size :', len(tokenizer)); print('PAD        :', tokenizer.pad_token_id); print('BOS        :', tokenizer.bos_token_id); print('EOS        :', tokenizer.eos_token_id); print('UNK        :', tokenizer.unk_token_id); print('Text       :', text); print('IDs        :', enc['input_ids']); print('Tokens     :', tokens); print('Decode     :', tokenizer.decode(enc['input_ids'], skip_special_tokens=False)); print('UNK number :', unk_num); print('=' * 70 + '\n')
    assert len(tokenizer) > 1000, f'Abnormal tokenizer vocab size: {len(tokenizer)}'; assert unk_num == 0, f'Tokenizer contains <unk>: {tokens}'

def save_model(accelerator, model, tokenizer, save_path):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and os.path.isdir(save_path): shutil.rmtree(save_path)
    accelerator.wait_for_everyone(); unwrapped_model = accelerator.unwrap_model(model); unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    if accelerator.is_main_process: tokenizer.save_pretrained(save_path)
    accelerator.wait_for_everyone()

def trainval(cfg):
    init_nltk(); checkpoint_path = os.path.join(cfg.checkpoint_path, cfg.checkpoint_name); best_model_path = os.path.join(checkpoint_path, 'best_model'); last_model_path = os.path.join(checkpoint_path, 'last_model')
    accelerator_log_kwargs = {}
    if cfg.use_logger: accelerator_log_kwargs['log_with'] = cfg.logger_type; accelerator_log_kwargs['project_dir'] = checkpoint_path
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, mixed_precision='bf16', **accelerator_log_kwargs)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO); logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process: datasets.utils.logging.set_verbosity_warning(); transformers.utils.logging.set_verbosity_warning()
    else: datasets.utils.logging.set_verbosity_error(); transformers.utils.logging.set_verbosity_error()
    if cfg.seed is not None: set_seed(cfg.seed)
    if accelerator.is_main_process: os.makedirs(checkpoint_path, exist_ok=True)
    accelerator.wait_for_everyone()

    preprocessed_train_dataset_name = f'{cfg.dataset_name}-train-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}-multimodal.json'; preprocessed_val_dataset_name = f'{cfg.dataset_name}-val-{cfg.obs_len}-{cfg.pred_len}-{cfg.metric}.json'; preprocessed_dataset_path = os.path.join(cfg.dataset_path, 'preprocessed')
    data_files = {'train': os.path.join(preprocessed_dataset_path, preprocessed_train_dataset_name), 'validation': os.path.join(preprocessed_dataset_path, preprocessed_val_dataset_name)}
    print('Train dataset:', data_files['train']); print('Validation dataset:', data_files['validation'])
    if not os.path.exists(data_files['train']) or not os.path.exists(data_files['validation']): raise ValueError(f"Preprocessed dataset files not found:\n{data_files['train']}\n{data_files['validation']}")
    extension = data_files['train'].split('.')[-1]; raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cfg.cache_dir)

    if cfg.model_name_or_path: print('\nLoading LOCAL Qwen config:', cfg.model_name_or_path); config = AutoConfig.from_pretrained(cfg.model_name_or_path, trust_remote_code=False, cache_dir=cfg.cache_dir, local_files_only=True)
    elif cfg.model_config_name: config = AutoConfig.from_pretrained(cfg.model_config_name, trust_remote_code=False, cache_dir=cfg.cache_dir, local_files_only=True)
    else:
        if cfg.model_type is None: raise ValueError('model_type cannot be None when training from scratch.')
        config = CONFIG_MAPPING[cfg.model_type]()

    print('\n' + '=' * 70); print('Config class:', config.__class__.__name__); print('Model type  :', config.model_type); print('=' * 70)

    if cfg.tokenizer_name: print('\nLoading trajectory tokenizer:', cfg.tokenizer_name); tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True, local_files_only=True)
    elif cfg.model_name_or_path: tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=False, cache_dir=cfg.cache_dir, use_fast=True, local_files_only=True)
    else: raise ValueError('Tokenizer is not specified.')

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None: raise ValueError('Tokenizer has neither PAD nor EOS token.')
        tokenizer.pad_token = tokenizer.eos_token
    check_tokenizer(tokenizer)

    if cfg.model_name_or_path: print('\nLoading local pretrained Qwen model:', cfg.model_name_or_path); model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, config=config, trust_remote_code=False, cache_dir=cfg.cache_dir, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    else: logger.info('Training model from scratch'); config.vocab_size = len(tokenizer); model = AutoModelForCausalLM.from_config(config, trust_remote_code=False); model = model.to(dtype=torch.bfloat16)

    print('\n' + '=' * 70); print('Model class:', model.__class__.__name__); print('Model dtype:', next(model.parameters()).dtype); print('=' * 70)
    model = adapt_model_to_tokenizer(model, tokenizer); model.config.use_cache = False; model.gradient_checkpointing_enable()

    print('\n===== FINAL TOKENIZER / MODEL CHECK ====='); print('Tokenizer vocab :', len(tokenizer)); print('Model embedding :', model.get_input_embeddings().num_embeddings); print('Config vocab    :', model.config.vocab_size); print('Model dtype     :', next(model.parameters()).dtype)
    assert model.get_input_embeddings().num_embeddings == len(tokenizer); assert model.config.vocab_size == len(tokenizer)

    total_parameters = sum(p.numel() for p in model.parameters()); trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n===== MODEL PARAMETERS ====='); print(f'Total parameters     : {total_parameters:,}'); print(f'Total parameters (M) : {total_parameters / 1e6:.3f} M'); print(f'Trainable parameters : {trainable_parameters:,}'); print(f'Trainable ratio       : {100 * trainable_parameters / total_parameters:.2f}%'); print('============================\n')
    assert trainable_parameters == total_parameters

    column_names = raw_datasets['train'].column_names; history_column = cfg.history_column; future_column = cfg.future_column
    if history_column not in column_names: raise ValueError(f'{history_column} not found in dataset.')
    if future_column not in column_names: raise ValueError(f'{future_column} not found in dataset.')

    def encode_source(text): return tokenizer(text, max_length=cfg.max_source_length, truncation=True, add_special_tokens=False)['input_ids']

    def encode_target(text):
        max_target_length = cfg.max_target_length - 1 if tokenizer.eos_token_id is not None else cfg.max_target_length; ids = tokenizer(text, max_length=max_target_length, truncation=True, add_special_tokens=False)['input_ids']
        if tokenizer.eos_token_id is not None: ids = ids + [tokenizer.eos_token_id]
        return ids

    def preprocess_train_function(examples):
        model_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for source, target in zip(examples[history_column], examples[future_column]):
            source_ids = encode_source(source); target_ids = encode_target(target); input_ids = source_ids + target_ids; labels = [-100] * len(source_ids) + target_ids.copy(); model_inputs['input_ids'].append(input_ids); model_inputs['attention_mask'].append([1] * len(input_ids)); model_inputs['labels'].append(labels)
        return model_inputs

    def preprocess_val_function(examples):
        model_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for source, target in zip(examples[history_column], examples[future_column]):
            source_ids = encode_source(source); target_ids = encode_target(target); model_inputs['input_ids'].append(source_ids); model_inputs['attention_mask'].append([1] * len(source_ids)); model_inputs['labels'].append(target_ids)
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets['train'].map(preprocess_train_function, batched=True, remove_columns=column_names, load_from_cache_file=False, desc='Tokenizing training dataset')
        val_dataset = raw_datasets['validation'].map(preprocess_val_function, batched=True, remove_columns=column_names, load_from_cache_file=False, desc='Tokenizing validation dataset')

    index = random.randint(0, len(train_dataset) - 1); print('\n===== TOKENIZED TRAINING SAMPLE ====='); print('Index:', index); sample_ids = train_dataset[index]['input_ids'][:50]; print('First 50 IDs   :', sample_ids); print('First 50 tokens:', tokenizer.convert_ids_to_tokens(sample_ids)); print('Decoded input  :', tokenizer.decode(train_dataset[index]['input_ids'], skip_special_tokens=True)); valid_labels = [x for x in train_dataset[index]['labels'] if x != -100]; print('Decoded target :', tokenizer.decode(valid_labels, skip_special_tokens=True)); print('=====================================\n')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, label_pad_token_id=-100, pad_to_multiple_of=8)
    train_batch_size = 4; eval_batch_size = 4
    print('\n===== RTX 5070 8GB SETTINGS ====='); print('Actual train batch size :', train_batch_size); print('Actual eval batch size  :', eval_batch_size); print('Gradient accumulation   :', cfg.gradient_accumulation_steps); print('Effective batch size    :', train_batch_size * cfg.gradient_accumulation_steps); print('=================================\n')

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size, pin_memory=True)
    eval_dataloader = DataLoader(val_dataset, shuffle=False, collate_fn=data_collator, batch_size=eval_batch_size, pin_memory=True)

    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.weight_decay}, {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    try: optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, fused=True); print('Optimizer: AdamW (fused=True)')
    except Exception as e: print('Fused AdamW unavailable, fallback to normal AdamW.'); print('Reason:', e); optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None: cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    else: cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(name=cfg.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=cfg.max_train_steps)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    print('\n' + '=' * 70); print('DEVICE / PRECISION CHECK'); print('Accelerator device :', accelerator.device); print('Model device       :', next(model.parameters()).device); print('Model dtype        :', next(model.parameters()).dtype); print('Mixed precision    :', accelerator.mixed_precision); print('CUDA available     :', torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU                :', torch.cuda.get_device_name(0)); print('GPU capability     :', torch.cuda.get_device_capability(0)); print(f'GPU allocated GB   : {torch.cuda.memory_allocated(0) / 1024**3:.3f}'); print(f'GPU reserved GB    : {torch.cuda.memory_reserved(0) / 1024**3:.3f}')
    print('=' * 70 + '\n')
    if accelerator.device.type != 'cuda': raise RuntimeError('Training is not running on CUDA.')

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps); cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    if cfg.use_logger: accelerator.init_trackers('Language-Based Trajectory Predictor')

    total_batch_size = train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    logger.info('***** Running training *****'); logger.info(f'Num examples = {len(train_dataset)}'); logger.info(f'Num Epochs = {cfg.num_train_epochs}'); logger.info(f'Instantaneous batch size = {train_batch_size}'); logger.info(f'Total train batch size = {total_batch_size}'); logger.info(f'Gradient Accumulation steps = {cfg.gradient_accumulation_steps}'); logger.info(f'Total optimization steps = {cfg.max_train_steps}')

    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process); completed_steps = 0; best_metric_name = getattr(cfg, 'best_metric', None) or 'rougeLsum'; valid_metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    if best_metric_name not in valid_metrics: raise ValueError(f'best_metric must be one of {valid_metrics}, but got {best_metric_name}')
    best_score, best_epoch, best_result, last_result = -float('inf'), -1, None, None

    for epoch in range(cfg.num_train_epochs):
        model.train(); total_train_loss = torch.tensor(0.0, device=accelerator.device); train_batch_count = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch); loss = outputs.loss; total_train_loss += loss.detach().float(); train_batch_count += 1; accelerator.backward(loss); optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients: progress_bar.update(1); completed_steps += 1
            if completed_steps >= cfg.max_train_steps: break

        train_loss_tensor = total_train_loss / max(train_batch_count, 1); train_loss = accelerator.gather(train_loss_tensor.reshape(1)).mean().item()

        model.eval(); rouge_metric = evaluate.load('rouge')
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                input_length = batch['input_ids'].shape[1]; generated_tokens = accelerator.unwrap_model(model).generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_new_tokens=cfg.max_target_length, num_beams=cfg.num_beams, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True); generated_tokens = generated_tokens[:, input_length:]
            generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id); labels = accelerator.pad_across_processes(batch['labels'], dim=1, pad_index=-100); generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels)); generated_tokens = generated_tokens.detach().cpu().numpy(); labels = labels.detach().cpu().numpy(); labels = np.where(labels != -100, labels, tokenizer.pad_token_id); decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False); decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False); decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels); rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        raw_result = rouge_metric.compute(use_stemmer=True); result = {k: round(float(v) * 100, 4) for k, v in raw_result.items()}; last_result = result.copy(); current_score = float(result[best_metric_name])

        accelerator.print('\n' + '=' * 70); accelerator.print(f'Epoch {epoch + 1}/{cfg.num_train_epochs}'); accelerator.print(f'Train Loss   : {train_loss:.6f}'); accelerator.print(f"ROUGE-1      : {result['rouge1']:.4f}"); accelerator.print(f"ROUGE-2      : {result['rouge2']:.4f}"); accelerator.print(f"ROUGE-L      : {result['rougeL']:.4f}"); accelerator.print(f"ROUGE-Lsum   : {result['rougeLsum']:.4f}"); accelerator.print(f'Best metric  : {best_metric_name}'); accelerator.print('=' * 70)

        if current_score > best_score:
            best_score, best_epoch = current_score, epoch + 1; best_result = {k: float(v) for k, v in result.items()}; accelerator.print('\n*** New Best Model ***'); accelerator.print(f'Epoch = {best_epoch}'); accelerator.print(f'{best_metric_name} = {best_score:.4f}'); save_model(accelerator, model, tokenizer, best_model_path)
            if accelerator.is_main_process:
                best_info = {'best_epoch': int(best_epoch), 'best_metric': best_metric_name, 'best_score': float(best_score), 'train_loss': float(train_loss), 'rouge1': float(result['rouge1']), 'rouge2': float(result['rouge2']), 'rougeL': float(result['rougeL']), 'rougeLsum': float(result['rougeLsum'])}
                with open(os.path.join(best_model_path, 'best_results.json'), 'w', encoding='utf-8') as f: json.dump(best_info, f, ensure_ascii=False, indent=4)

        accelerator.print(f'\nCurrent Best -> Epoch: {best_epoch}, {best_metric_name}: {best_score:.4f}')
        if cfg.use_logger: accelerator.log({'train_loss': float(train_loss), 'rouge1': float(result['rouge1']), 'rouge2': float(result['rouge2']), 'rougeL': float(result['rougeL']), 'rougeLsum': float(result['rougeLsum']), 'epoch': int(epoch + 1)}, step=completed_steps)
        torch.cuda.empty_cache()

    accelerator.print('\nSaving LAST model...'); save_model(accelerator, model, tokenizer, last_model_path)

    if accelerator.is_main_process:
        last_info = {'epoch': int(cfg.num_train_epochs), 'rouge1': float(last_result['rouge1']), 'rouge2': float(last_result['rouge2']), 'rougeL': float(last_result['rougeL']), 'rougeLsum': float(last_result['rougeLsum'])}
        with open(os.path.join(last_model_path, 'last_results.json'), 'w', encoding='utf-8') as f: json.dump(last_info, f, ensure_ascii=False, indent=4)
        summary = {'best_epoch': int(best_epoch), 'best_metric': best_metric_name, 'best_score': float(best_score), 'best_result': best_result, 'last_result': last_result}
        with open(os.path.join(checkpoint_path, 'training_summary.json'), 'w', encoding='utf-8') as f: json.dump(summary, f, ensure_ascii=False, indent=4)

    accelerator.print('\n' + '=' * 70); accelerator.print('Training Finished'); accelerator.print(f'Best Epoch      : {best_epoch}'); accelerator.print(f'Best Metric     : {best_metric_name}'); accelerator.print(f'Best Score      : {best_score:.4f}'); accelerator.print(f'Best Model Path : {best_model_path}'); accelerator.print(f'Last Model Path : {last_model_path}'); accelerator.print('=' * 70)

import argparse
from utils.config import get_exp_config, print_arguments

parser = argparse.ArgumentParser(); parser.add_argument('--cfg', default='./config/config-meter.json', type=str, help='Config file path.'); parser.add_argument('--dataset', default='ZSI', type=str, help='Dataset name.'); parser.add_argument('--tag', default='LMTraj_Qwen35', type=str, help='Checkpoint name.'); parser.add_argument('--test', default=False, action='store_true', help='Evaluation mode.'); args = parser.parse_args([])

print('===== Arguments ====='); print_arguments(vars(args)); print('===== Configs ====='); cfg = get_exp_config(args.cfg); print_arguments(cfg); cfg.dataset_name = args.dataset; cfg.checkpoint_name = args.tag
print('\nTransformers version:', transformers.__version__); print('PyTorch version     :', torch.__version__); print('PyTorch CUDA        :', torch.version.cuda); print('CUDA available      :', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU                 :', torch.cuda.get_device_name(0)); print('GPU capability      :', torch.cuda.get_device_capability(0)); print('GPU total memory GB :', torch.cuda.get_device_properties(0).total_memory / 1024**3)
print('Current working directory:', os.getcwd()); print('\nModel path:', cfg.model_name_or_path); print('Model directory exists:', os.path.isdir(cfg.model_name_or_path)); print('\nTokenizer path:', cfg.tokenizer_name); print('Tokenizer directory exists:', os.path.isdir(cfg.tokenizer_name))
if not os.path.isdir(cfg.model_name_or_path): raise FileNotFoundError(f'Model directory not found: {cfg.model_name_or_path}')
if not os.path.isfile(os.path.join(cfg.model_name_or_path, 'config.json')): raise FileNotFoundError('config.json not found.')
weight_files = [f for f in os.listdir(cfg.model_name_or_path) if f.endswith('.safetensors') or f.endswith('.bin')]
if len(weight_files) == 0: raise FileNotFoundError('No model weight file (.safetensors/.bin) found.')
print('\nModel weight files:')
for f in weight_files: print(' ', f)
if not os.path.isdir(cfg.tokenizer_name): raise FileNotFoundError(f'Tokenizer directory not found: {cfg.tokenizer_name}')
print('\nAll paths checked successfully.')
trainval(cfg)