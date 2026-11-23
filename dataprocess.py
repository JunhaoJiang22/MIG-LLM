
import os
import json
import copy
import argparse
import numpy as np
from tqdm import tqdm
from utils.dataloader import get_dataloader
from utils.converter import batch_traj2txt, change_template
from utils.utils import reproducibility_settings, round_floats

# Prompt templates
HIST_VESSEL_TEMPLATE = "Vessl {} moved along the trajectory {} for {} frames."
QUESTION_ANSWER_TEMPLATE = "question: {} context: {} answer:"
QUESTION_TEMPLATE = {"forecast": "What trajectory does vessel {} follow for the next {} frames?",
                     "destination": "At which coordinates does vessel {} arrive after the next {} frames?",
                     "direction": "In which direction will vessel {} move in the future?", 
                     "speed": "What is the expected speed of vessel {} over the next {} frames?", 
                     "collision": "With which vessel does vessel {} have a collision risk?"}
ANSWER_TEMPLATE = {"forecast": "Vessl {} will move along the trajectory {} for the next {} frames.", 
                   "destination": "Vessl {} will arrive at coordinate {} after the next {} frames.", 
                   "direction": "Vessl {} will {}.", "speed": "Vessel {} is expected to move at {} speed.", 
                   "collision": "Vessl {} has a collision risk possibility with vessel {}.", 
                   "collision_false": "Vessl {} has no collision risk.", 
                   "direction_answer_list": ["maintain course", "move astern", "turn to port", "turn to starboard", "nearly stationary"], 
                   "speed_answer_list": ["low", "moderate", "high", "nearly stationary"]}
IMAGE_SCALE_DOWN = 0.25
AUGMENTATION_SHIFT = [-2, -1, 0, 1, 2]
NUMPED_THRESHOLD = 5
COORD_PRECISION = {"meter": "{:.2f}", "pixel": "{:d}"}


# Auxiliary-task labels
def get_direction_type(obs_traj, pred_traj, target_idx=0):
    DIRECTION_RADIUS_THRESHOLD = 20
    DIRECTION_BACKWARD_THRESHOLD = 90
    DIRECTION_FORWARD_THRESHOLD = 5
    obs_len = obs_traj.shape[1]
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)[target_idx]
    full_traj_disp = full_traj[1:] - full_traj[:-1]
    # Preserve the original static-vessel rule
    if np.linalg.norm(full_traj[obs_len] - full_traj[pred_traj.shape[1]], ord=2) < DIRECTION_RADIUS_THRESHOLD:
        return 4
    direction = full_traj[obs_len] - full_traj[obs_len - 2]
    rot = np.arctan2(direction[1], direction[0])
    traj_rot = np.array([[np.cos(rot), np.sin(-rot)], [np.sin(rot), np.cos(rot)]])
    full_traj_disp_norm = full_traj_disp @ traj_rot
    future_norm = full_traj_disp_norm[obs_len:].mean(axis=0)
    future_dir = np.arctan2(future_norm[1], future_norm[0]) * 180.0 / np.pi
    if future_dir < -DIRECTION_BACKWARD_THRESHOLD or future_dir > DIRECTION_BACKWARD_THRESHOLD:
        return 1
    if future_dir > DIRECTION_FORWARD_THRESHOLD:
        return 2
    if future_dir < -DIRECTION_FORWARD_THRESHOLD:
        return 3
    return 0


def get_speed_type(obs_traj, pred_traj, target_idx=0):
    SPEED_STOP_THRESHOLD = 2.0
    SPEED_LOW_THRESHOLD = 8.0
    SPEED_HIGH_THRESHOLD = 15.0
    obs_len = obs_traj.shape[1]
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)[target_idx]
    full_traj_disp = full_traj[1:] - full_traj[:-1]
    future_disp = full_traj_disp[obs_len - 1:]
    # Sampling interval: 5 s, conversion: 1 knot = 0.5144 m/s
    mean_speed = np.linalg.norm(future_disp, ord=2, axis=-1).mean() / 5.0 / 0.5144
    if mean_speed < SPEED_STOP_THRESHOLD:
        return 3
    if mean_speed < SPEED_LOW_THRESHOLD:
        return 0
    if mean_speed < SPEED_HIGH_THRESHOLD:
        return 1
    return 2


# CPA computation
def compute_dcpa_tcpa(ref_state, tar_state, eps=1e-10):
    """Input: [lon(deg), lat(deg), SOG(kn), COG(deg)]. Output: DCPA(nm), TCPA(min)."""
    ref_lon, ref_lat, ref_sog, ref_cog = map(float, ref_state)
    tar_lon, tar_lat, tar_sog, tar_cog = map(float, tar_state)
    vals = [ref_lon, ref_lat, ref_sog, ref_cog, tar_lon, tar_lat, tar_sog, tar_cog]
    if not np.all(np.isfinite(vals)):
        return np.inf, np.inf
    mean_lat = np.deg2rad((ref_lat + tar_lat) * 0.5)
    d_east_nm = (tar_lon - ref_lon) * 60.0 * np.cos(mean_lat)
    d_north_nm = (tar_lat - ref_lat) * 60.0
    r = np.array([d_east_nm, d_north_nm], dtype=np.float64)
    ref_cog_rad = np.deg2rad(ref_cog)
    tar_cog_rad = np.deg2rad(tar_cog)
    v_ref = np.array([ref_sog * np.sin(ref_cog_rad), ref_sog * np.cos(ref_cog_rad)], dtype=np.float64)
    v_tar = np.array([tar_sog * np.sin(tar_cog_rad), tar_sog * np.cos(tar_cog_rad)], dtype=np.float64)
    v_rel = v_tar - v_ref
    rel_speed_sq = float(np.dot(v_rel, v_rel))
    current_dist = float(np.linalg.norm(r))
    if rel_speed_sq < eps:
        return current_dist, np.inf
    tcpa_hour = -float(np.dot(r, v_rel)) / rel_speed_sq
    closest_r = r + v_rel * tcpa_hour
    dcpa_nm = float(np.linalg.norm(closest_r))
    tcpa_min = tcpa_hour * 60.0
    return dcpa_nm, tcpa_min


def get_collision_id(pred_traj_cpa, target_idx, dcpa_threshold=0.5, tcpa_threshold=5.0, risk_logic="or"):
    """Return the local context ID of the selected risky vessel."""
    num_ship, pred_len, feat_dim = pred_traj_cpa.shape
    if feat_dim < 4:
        raise ValueError("pred_traj_CPA must have [lon, lat, sog, cog].")
    if num_ship <= 1:
        return None
    if not (0 <= target_idx < num_ship):
        raise IndexError(f"target_idx={target_idx}, num_ship={num_ship}")
    neighbor_ids = [i for i in range(num_ship) if i != target_idx]
    dcpa_all = np.full((len(neighbor_ids), pred_len), np.inf, dtype=np.float64)
    tcpa_all = np.full((len(neighbor_ids), pred_len), np.inf, dtype=np.float64)
    for ni, neighbor_idx in enumerate(neighbor_ids):
        for t in range(pred_len):
            dcpa, tcpa = compute_dcpa_tcpa(pred_traj_cpa[target_idx, t], pred_traj_cpa[neighbor_idx, t])
            dcpa_all[ni, t] = abs(dcpa)
            tcpa_all[ni, t] = tcpa
    dcpa_risk = dcpa_all <= dcpa_threshold
    tcpa_risk = np.isfinite(tcpa_all) & (tcpa_all >= 0.0) & (tcpa_all <= tcpa_threshold)
    if risk_logic.lower() == "or":
        risk_per_time = dcpa_risk | tcpa_risk
    elif risk_logic.lower() == "and":
        risk_per_time = dcpa_risk & tcpa_risk
    else:
        raise ValueError("risk_logic must be 'or' or 'and'.")
    risk_mask = np.any(risk_per_time, axis=1)
    min_dcpa = np.min(dcpa_all, axis=1)
    order = np.argsort(min_dcpa)
    for pos in order:
        if risk_mask[pos]:
            return neighbor_ids[pos]
    return None


# Context vessel selection
def select_context_indices(obs_traj_rel, target_global_idx, max_peds=5):
    """Select the target and nearest neighbors, preserving original scene order."""
    target = obs_traj_rel[target_global_idx]
    distances = np.linalg.norm(obs_traj_rel - target[None, :, :], ord=2, axis=-1)[:, -1]
    nearest = np.argsort(distances)
    neighbors = [int(i) for i in nearest if int(i) != int(target_global_idx)]
    neighbors = neighbors[:max(0, max_peds - 1)]
    selected_global = sorted([int(target_global_idx)] + neighbors)
    target_local_idx = selected_global.index(int(target_global_idx))
    return np.asarray(selected_global, dtype=np.int64), target_local_idx


# Main preprocessing
def process_dataset(dataset, phase, obs_len=20, pred_len=12, coord_system="meter", 
                    use_scene_context=True, multimodal=None, augment=None, dcpa_threshold=0.5, tcpa_threshold=5.0, risk_logic="or", output=None):
    if multimodal is None:
        multimodal = ["forecast", "destination", "direction", "speed", "collision"]
    if augment is None:
        augment = []
    if phase not in ["train", "val", "test"]:
        raise ValueError("phase must be train/val/test")
    if phase != "train":
        augment = []
    data_dir = f"./datasets/{dataset}/Mo"
    data_dir_cpa = f"./datasets/{dataset}/Lng"
    if output is None:
        postfix = "-multimodal" if len(multimodal) > 1 else ""
        output = f"./datasets/preprocessed/{dataset}-{phase}-{obs_len}-{pred_len}-{coord_system}{postfix}.json"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    reproducibility_settings()
    dataloader = get_dataloader(data_dir, data_dir_cpa, phase, obs_len, pred_len, batch_size=1)
    scene_img = dataloader.dataset.scene_img
    scene_desc = dataloader.dataset.scene_desc
    if coord_system == "meter":
        change_template({"coord_template": COORD_PRECISION[coord_system]})
    else:
        raise NotImplementedError("This corrected script currently targets meter coordinates.")
    processed_data = {m: [] for m in multimodal}
    collision_positive = 0
    collision_total = 0

    for batch in tqdm(dataloader, desc=f"Processing {dataset} {phase}"):
        obs_traj_rel = batch["obs_traj_rel"].numpy()
        pred_traj_rel = batch["pred_traj_rel"].numpy()
        obs_traj_abs = batch["obs_traj"].numpy()
        pred_traj_abs = batch["pred_traj"].numpy()
        pred_traj_cpa = batch["pred_traj_CPA"].numpy()
        non_linear_ped = batch["non_linear_ped"].numpy()
        frame = batch["frame"].numpy()
        scene_id = batch["scene_id"]
        vessel_ids = batch["vessel_id"].numpy()
        n_ped = obs_traj_rel.shape[0]

        aug_param = []
        if "shift" in augment:
            for x_shift in AUGMENTATION_SHIFT:
                for y_shift in AUGMENTATION_SHIFT:
                    aug_param.append({"shift": np.array([x_shift, y_shift]), "flip": False, "swap": False, "reverse": False})
        else:
            aug_param.append({"shift": np.array([0, 0]), "flip": False, "swap": False, "reverse": False})
        if "flip" in augment:
            temp = copy.deepcopy(aug_param)
            for ap in temp:
                ap["flip"] = True
            aug_param.extend(temp)
        if "swap" in augment:
            temp = copy.deepcopy(aug_param)
            for ap in temp:
                ap["swap"] = True
            aug_param.extend(temp)
        if "reverse" in augment:
            if "collision" in multimodal:
                raise ValueError("Do not use reverse augmentation with collision CPA labels.")
            temp = copy.deepcopy(aug_param)
            for ap in temp:
                ap["reverse"] = True
            aug_param.extend(temp)

        for target_global_idx in range(n_ped):
            selected, target_local_idx = select_context_indices(obs_traj_rel, target_global_idx, max_peds=NUMPED_THRESHOLD)
            obs_rel_ctx = obs_traj_rel[selected]
            pred_rel_ctx = pred_traj_rel[selected]
            obs_abs_ctx = obs_traj_abs[selected]
            pred_abs_ctx = pred_traj_abs[selected]
            pred_cpa_ctx = pred_traj_cpa[selected]
            vessel_ids_ctx = vessel_ids[selected]
            n_ped_ctx = len(selected)
            direction_type = get_direction_type(obs_abs_ctx, pred_abs_ctx, target_local_idx) if "direction" in multimodal else None
            speed_type = get_speed_type(obs_abs_ctx, pred_abs_ctx, target_local_idx) if "speed" in multimodal else None
            collision_local_id = get_collision_id(pred_cpa_ctx, target_local_idx, dcpa_threshold=dcpa_threshold, 
                                                  tcpa_threshold=tcpa_threshold, risk_logic=risk_logic) if "collision" in multimodal else None
            if "collision" in multimodal:
                collision_total += 1
                collision_positive += int(collision_local_id is not None)

            for ap in aug_param:
                obs_aug = obs_rel_ctx.copy()
                pred_aug = pred_rel_ctx.copy()
                obs_aug += ap["shift"][None, None, :]
                pred_aug += ap["shift"][None, None, :]
                if ap["flip"]:
                    if coord_system == "pixel":
                        img = scene_img.get(scene_id[target_global_idx])
                        if img is None:
                            raise RuntimeError("Pixel flip requested but scene image is unavailable.")
                        scene_img_size = (np.array(img.size)[None, None, :] * IMAGE_SCALE_DOWN).astype(np.int32)
                    else:
                        mid = [13518287.2567, 3696502.6884]
                        scene_img_size = np.array(mid)[None, None, :]
                    obs_aug = -obs_aug + scene_img_size
                    pred_aug = -pred_aug + scene_img_size
                if ap["swap"]:
                    obs_aug = obs_aug[:, :, [1, 0]]
                    pred_aug = pred_aug[:, :, [1, 0]]
                if ap["reverse"]:
                    full_aug = np.concatenate([obs_aug, pred_aug], axis=1)[:, ::-1]
                    obs_aug = full_aug[:, :obs_len]
                    pred_aug = full_aug[:, obs_len:]

                scene_context = scene_desc.get(scene_id[target_global_idx], "")
                obs_text = batch_traj2txt(obs_aug)
                pred_text = batch_traj2txt(pred_aug)
                traj_context = " ".join(HIST_VESSEL_TEMPLATE.format(i, obs_text[i], obs_len) for i in range(n_ped_ctx))
                context = (traj_context + " " + scene_context).strip() if use_scene_context else traj_context
                meta = {"scene": scene_id[target_global_idx], "frame": int(frame[target_global_idx]), "n_ped": int(n_ped), "n_ped_context": int(n_ped_ctx), "ped_id": int(target_local_idx), "ped_id_global": int(target_global_idx), "vessel_id": int(vessel_ids[target_global_idx]), "context_vessel_ids": [int(v) for v in vessel_ids_ctx.tolist()], "non_linear_ped": non_linear_ped[target_global_idx].tolist()}

                if "forecast" in multimodal:
                    question = QUESTION_TEMPLATE["forecast"].format(target_local_idx, pred_len)
                    obs_prompt = QUESTION_ANSWER_TEMPLATE.format(question, context)
                    pred_prompt = ANSWER_TEMPLATE["forecast"].format(target_local_idx, pred_text[target_local_idx], pred_len)
                    processed_data["forecast"].append({"id": len(processed_data["forecast"]), "type": "forecast", **meta, "obs_traj": obs_aug[target_local_idx].tolist(), "pred_traj": pred_aug[target_local_idx].tolist(), "observation": obs_prompt, "forecast": pred_prompt})

                # Auxiliary tasks use absolute trajectories
                aux_obs_text = batch_traj2txt(obs_rel_ctx)
                aux_traj_context = " ".join(HIST_VESSEL_TEMPLATE.format(i, aux_obs_text[i], obs_len) for i in range(n_ped_ctx))
                abs_context = (scene_context + " " + aux_traj_context).strip() if use_scene_context else aux_traj_context

                if "destination" in multimodal:
                    question = QUESTION_TEMPLATE["destination"].format(target_local_idx, pred_len)
                    obs_prompt = QUESTION_ANSWER_TEMPLATE.format(question, abs_context)
                    dest = pred_abs_ctx[target_local_idx, -1]
                    dest_coord = "(" + ", ".join(COORD_PRECISION[coord_system].format(dest[j]) for j in range(2)) + ")"
                    pred_prompt = ANSWER_TEMPLATE["destination"].format(target_local_idx, dest_coord, pred_len)
                    processed_data["destination"].append({"id": len(processed_data["destination"]), "type": "destination", **meta, "obs_traj": obs_abs_ctx[target_local_idx].tolist(), "pred_traj": pred_abs_ctx[target_local_idx].tolist(), "observation": obs_prompt, "forecast": pred_prompt})

                if "direction" in multimodal:
                    question = QUESTION_TEMPLATE["direction"].format(target_local_idx)
                    obs_prompt = QUESTION_ANSWER_TEMPLATE.format(question, abs_context)
                    direction = ANSWER_TEMPLATE["direction_answer_list"][direction_type]
                    pred_prompt = ANSWER_TEMPLATE["direction"].format(target_local_idx, direction)
                    processed_data["direction"].append({"id": len(processed_data["direction"]), "type": "direction", **meta, "obs_traj": obs_abs_ctx[target_local_idx].tolist(), "pred_traj": pred_abs_ctx[target_local_idx].tolist(), "observation": obs_prompt, "forecast": pred_prompt})

                if "speed" in multimodal:
                    question = QUESTION_TEMPLATE["speed"].format(target_local_idx, pred_len)
                    obs_prompt = QUESTION_ANSWER_TEMPLATE.format(question, abs_context)
                    speed = ANSWER_TEMPLATE["speed_answer_list"][speed_type]
                    pred_prompt = ANSWER_TEMPLATE["speed"].format(target_local_idx, speed)
                    processed_data["speed"].append({"id": len(processed_data["speed"]), "type": "speed", **meta, "obs_traj": obs_abs_ctx[target_local_idx].tolist(), "pred_traj": pred_abs_ctx[target_local_idx].tolist(), "observation": obs_prompt, "forecast": pred_prompt})

                if "collision" in multimodal:
                    question = QUESTION_TEMPLATE["collision"].format(target_local_idx)
                    obs_prompt = QUESTION_ANSWER_TEMPLATE.format(question, abs_context)
                    if collision_local_id is not None:
                        pred_prompt = ANSWER_TEMPLATE["collision"].format(target_local_idx, collision_local_id)
                    else:
                        pred_prompt = ANSWER_TEMPLATE["collision_false"].format(target_local_idx)
                    processed_data["collision"].append({"id": len(processed_data["collision"]), 
                                                        "type": "collision", **meta, 
                                                        "collision_local_id": int(collision_local_id) if collision_local_id is not None else None, 
                                                        "obs_traj": obs_abs_ctx[target_local_idx].tolist(), 
                                                        "pred_traj": pred_abs_ctx[target_local_idx].tolist(), "observation": obs_prompt, "forecast": pred_prompt})

    processed_data_cat = []
    for task in multimodal:
        processed_data_cat.extend(processed_data[task])
    with open(output, "w", encoding="utf-8") as f:
        for item in tqdm(processed_data_cat, desc=f"Writing {output}"):
            f.write(json.dumps(round_floats(item), ensure_ascii=False) + "\n")
    print(f"\nSaved: {output}")
    for task in multimodal:
        print(f"{task:12s}: {len(processed_data[task])}")
    if collision_total > 0:
        ratio = collision_positive / collision_total
        print(f"collision positive: {collision_positive}/{collision_total} \
              ({ratio:.2%}), DCPA<={dcpa_threshold} nm, TCPA<={tcpa_threshold} min, logic={risk_logic.upper()}")
    return processed_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ZSI")
    parser.add_argument("--phase", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--dcpa", type=float, default=0.5)
    parser.add_argument("--tcpa", type=float, default=5.0)
    parser.add_argument("--risk_logic", type=str, choices=["or", "and"], default="or")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=["forecast", "destination", "direction", "speed", "collision"], 
                        choices=["forecast", "destination", "direction", "speed", "collision"])
    return parser.parse_args([])


if __name__ == "__main__":
    args = parse_args()
    process_dataset(dataset=args.dataset, phase=args.phase, obs_len=args.obs_len, pred_len=args.pred_len, multimodal=args.tasks, 
                    dcpa_threshold=args.dcpa, tcpa_threshold=args.tcpa, risk_logic=args.risk_logic, output=args.output)