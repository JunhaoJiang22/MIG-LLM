import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
from PIL import Image

try:
    from .homography import generate_homography
except ImportError:
    generate_homography = None


def get_dataloader(data_dir, data_dir_cpa, phase, obs_len, pred_len, batch_size):
    """Mo: frame_id, vessel_id, x, y. Lng: frame_id, vessel_id, lon, lat, sog, cog."""
    assert phase in ["train", "val", "test"]
    data_set = os.path.join(data_dir, phase)
    data_set_cpa = os.path.join(data_dir_cpa, phase)
    shuffle = phase == "train"
    drop_last = phase == "train"
    dataset_phase = TrajectoryDataset(data_set, data_set_cpa, obs_len=obs_len, pred_len=pred_len)
    sampler_phase = None
    if batch_size > 1:
        sampler_phase = TrajBatchSampler(dataset_phase, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    loader_phase = DataLoader(dataset_phase, collate_fn=traj_collate_fn, batch_sampler=sampler_phase, pin_memory=True)
    return loader_phase


def traj_collate_fn(data):
    """Collate complete scenes while keeping Mo and CPA tensors aligned."""
    data_collated = {k: [d[k] for d in data] for k in data[0].keys()}
    lengths = [len(seq) for seq in data_collated["obs_traj"]]
    cum_start_idx = [0] + np.cumsum(lengths).tolist()
    seq_start_end = torch.LongTensor([[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])])
    scene_mask = torch.zeros(sum(lengths), sum(lengths), dtype=torch.bool)
    for start, end in seq_start_end:
        scene_mask[start:end, start:end] = True
    tensor_keys = ["obs_traj", "pred_traj", "obs_traj_rel", "pred_traj_rel", "obs_traj_CPA", "pred_traj_CPA", "non_linear_ped", "loss_mask", "frame", "vessel_id"]
    for key in tensor_keys:
        data_collated[key] = torch.cat(data_collated[key], dim=0)
    data_collated["scene_mask"] = scene_mask
    data_collated["seq_start_end"] = seq_start_end
    data_collated["scene_id"] = np.concatenate(data_collated["scene_id"], axis=0)
    return data_collated


class TrajBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=64, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        assert len(self.data_source) == len(self.data_source.num_peds_in_seq)
        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            else:
                generator = self.generator
            indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        else:
            indices = list(range(len(self.data_source)))
        num_peds_indices = self.data_source.num_peds_in_seq[indices]
        batch = []
        total_num_peds = 0
        for idx, num_peds in zip(indices, num_peds_indices):
            batch.append(idx)
            total_num_peds += int(num_peds)
            if total_num_peds >= self.batch_size:
                yield batch
                batch = []
                total_num_peds = 0
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total = int(sum(self.data_source.num_peds_in_seq))
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def read_file(path, delim="\t"):
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append([float(v) for v in line.split(delim)])
    return np.asarray(data, dtype=np.float64)


def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    err_x = float(res_x[0]) if len(res_x) else 0.0
    err_y = float(res_y[0]) if len(res_y) else 0.0
    return 1.0 if err_x + err_y >= threshold else 0.0


def _key(frame, vessel_id):
    """Stable matching key for the Mo/Lng files."""
    return round(float(frame), 8), round(float(vessel_id), 8)


def _file_map(directory):
    files = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            files[name] = path
    return files


class TrajectoryDataset(Dataset):
    """Match Mo and Lng rows by frame_id and vessel_id to keep trajectories aligned."""

    def __init__(self, data_dir, data_dir_cpa, obs_len=8, pred_len=12, skip=1, threshold=0.02, min_ped=1, delim="\t", strict_alignment=True):
        super().__init__()
        self.data_dir = data_dir
        self.data_dir_cpa = data_dir_cpa
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = obs_len + pred_len
        self.delim = delim
        self.strict_alignment = strict_alignment
        self.homography = {}
        self.scene_img = {}
        self.scene_map = {}
        self.scene_desc = {}
        mo_files = _file_map(self.data_dir)
        cpa_files = _file_map(self.data_dir_cpa)
        missing_cpa = sorted(set(mo_files) - set(cpa_files))
        missing_mo = sorted(set(cpa_files) - set(mo_files))
        if missing_cpa or missing_mo:
            raise RuntimeError(f"Mo/Lng file names are not aligned.\nMissing in Lng: {missing_cpa}\nMissing in Mo: {missing_mo}")

        seq_list = []
        seq_list_rel = []
        seq_list_cpa = []
        loss_mask_list = []
        non_linear_ped = []
        num_peds_in_seq = []
        frame_list = []
        scene_id = []
        vessel_id_list = []

        for filename in sorted(mo_files):
            mo_path = mo_files[filename]
            cpa_path = cpa_files[filename]
            data_mo = read_file(mo_path, delim)
            data_cpa = read_file(cpa_path, delim)
            if data_mo.ndim != 2 or data_mo.shape[1] < 4:
                raise ValueError(f"Mo file must have >=4 columns: {mo_path}")
            if data_cpa.ndim != 2 or data_cpa.shape[1] < 6:
                raise ValueError(f"Lng file must have >=6 columns [frame,id,lon,lat,sog,cog]: {cpa_path}")

            phase = os.path.basename(os.path.dirname(mo_path))
            scene_name = os.path.splitext(filename)[0]#.replace("_" + phase, "")
            #print(filename, phase, scene_name)
            self._load_scene_resources(mo_path, scene_name)

            # Match CPA records by frame and vessel identity
            cpa_lookup = {}
            for row in data_cpa:
                k = _key(row[0], row[1])
                if k in cpa_lookup:
                    raise RuntimeError(f"Duplicate (frame,vessel) in Lng file {cpa_path}: {k}")
                cpa_lookup[k] = row

            frames = np.unique(data_mo[:, 0]).tolist()
            max_start = len(frames) - self.seq_len
            if max_start < 0:
                continue

            for idx in range(0, max_start + 1, self.skip):
                window_frames = np.asarray(frames[idx:idx + self.seq_len], dtype=np.float64)
                frame_mask = np.isin(data_mo[:, 0], window_frames)
                curr_seq_data = data_mo[frame_mask]
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                mo_peds = []
                rel_peds = []
                cpa_peds = []
                masks = []
                nonlinear = []
                kept_vessel_ids = []

                for ped in peds_in_curr_seq:
                    ped_rows = curr_seq_data[curr_seq_data[:, 1] == ped]
                    ped_rows = ped_rows[np.argsort(ped_rows[:, 0])]
                    if len(ped_rows) != self.seq_len:
                        continue
                    if not np.allclose(ped_rows[:, 0], window_frames, atol=1e-8, rtol=0):
                        continue

                    xy = np.around(ped_rows[:, 2:4], decimals=4).T
                    cpa_rows = []
                    missing_keys = []
                    for fr in window_frames:
                        k = _key(fr, ped)
                        row = cpa_lookup.get(k)
                        if row is None:
                            missing_keys.append(k)
                        else:
                            cpa_rows.append(row)

                    if missing_keys:
                        msg = f"Mo/Lng alignment error in {filename}, vessel={ped}. Missing CPA rows, examples: {missing_keys[:5]}"
                        if self.strict_alignment:
                            raise RuntimeError(msg)
                        continue

                    cpa_rows = np.asarray(cpa_rows)
                    cpa = np.around(cpa_rows[:, 2:6], decimals=8).T
                    rel = np.zeros_like(xy)
                    rel[:, 1:] = xy[:, 1:] - xy[:, :-1]
                    mask = np.ones(self.seq_len, dtype=np.float64)
                    mo_peds.append(xy)
                    rel_peds.append(rel)
                    cpa_peds.append(cpa)
                    masks.append(mask)
                    nonlinear.append(poly_fit(xy, self.pred_len, threshold))
                    kept_vessel_ids.append(int(round(float(ped))))

                num_peds_considered = len(mo_peds)
                if num_peds_considered > min_ped:
                    mo_seq = np.stack(mo_peds, axis=0)
                    rel_seq = np.stack(rel_peds, axis=0)
                    cpa_seq = np.stack(cpa_peds, axis=0)
                    mask_seq = np.stack(masks, axis=0)
                    seq_list.append(mo_seq)
                    seq_list_rel.append(rel_seq)
                    seq_list_cpa.append(cpa_seq)
                    loss_mask_list.append(mask_seq)
                    non_linear_ped.extend(nonlinear)
                    num_peds_in_seq.append(num_peds_considered)
                    frame_list.extend([int(round(window_frames[0]))] * num_peds_considered)
                    scene_id.extend([scene_name] * num_peds_considered)
                    vessel_id_list.extend(kept_vessel_ids)

        if not seq_list:
            raise RuntimeError(f"No valid sequences found. Check obs_len/pred_len and Mo/Lng alignment. Mo={data_dir}, Lng={data_dir_cpa}")

        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_cpa = np.concatenate(seq_list_cpa, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.num_seq = len(num_peds_in_seq)
        self.num_peds_in_seq = np.asarray(num_peds_in_seq, dtype=np.int64)
        self.scene_id = np.asarray(scene_id)
        self.obs_traj = torch.from_numpy(seq_list[:, :, :obs_len]).float().permute(0, 2, 1)
        self.pred_traj = torch.from_numpy(seq_list[:, :, obs_len:]).float().permute(0, 2, 1)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :obs_len]).float().permute(0, 2, 1)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, obs_len:]).float().permute(0, 2, 1)
        self.obs_traj_CPA = torch.from_numpy(seq_list_cpa[:, :, :obs_len]).float().permute(0, 2, 1)
        self.pred_traj_CPA = torch.from_numpy(seq_list_cpa[:, :, obs_len:]).float().permute(0, 2, 1)
        self.loss_mask = torch.from_numpy(loss_mask_list).float().gt(0.5)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).float().gt(0.5)
        self.frame_list = torch.tensor(frame_list, dtype=torch.long)
        self.vessel_id = torch.tensor(vessel_id_list, dtype=torch.long)
        cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        if not (len(self.obs_traj) == len(self.obs_traj_CPA) == len(self.vessel_id) == int(np.sum(self.num_peds_in_seq))):
            raise RuntimeError("Internal Mo/Lng alignment check failed.")

    def _load_scene_resources(self, mo_path, scene_name):
        # ./datasets/ZSI/Mo/train/p_train.txt -> ./datasets/image
        phase_dir = os.path.dirname(mo_path)
        modality_dir = os.path.dirname(phase_dir)
        dataset_root = os.path.dirname(modality_dir)
        image_dir = os.path.join(os.path.dirname(dataset_root), "image")
        scene_img_map = {"p": "p_train", "p_train": "p_train", "p_test": "p_test", "p_val": "p_val"}
        image_stem = scene_img_map.get(scene_name)
        #print(scene_name, image_stem)
        if image_stem is None:
            self.scene_img[scene_name] = None
            self.scene_map[scene_name] = None
            self.scene_desc[scene_name] = ""
            return

        ref_path = os.path.join(image_dir, image_stem + "_reference.png")
        oracle_path = os.path.join(image_dir, image_stem + "_oracle.png")
        caption_path = os.path.join(image_dir, image_stem + "_caption.txt")
        try:
            self.scene_img[scene_name] = Image.open(ref_path)
        except Exception:
            self.scene_img[scene_name] = None
        try:
            self.scene_map[scene_name] = np.asarray(Image.open(oracle_path))
        except Exception:
            self.scene_map[scene_name] = None
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8-sig") as f:
                self.scene_desc[scene_name] = f.read().strip()
        else:
            self.scene_desc[scene_name] = ""

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return {"obs_traj": self.obs_traj[start:end], 
              "pred_traj": self.pred_traj[start:end], 
              "obs_traj_rel": self.obs_traj_rel[start:end], 
              "pred_traj_rel": self.pred_traj_rel[start:end], 
              "obs_traj_CPA": self.obs_traj_CPA[start:end], 
              "pred_traj_CPA": self.pred_traj_CPA[start:end], 
              "non_linear_ped": self.non_linear_ped[start:end], 
              "loss_mask": self.loss_mask[start:end], 
              "scene_mask": None, "seq_start_end": [[0, end - start]], 
              "frame": self.frame_list[start:end], 
              "scene_id": self.scene_id[start:end], 
              "vessel_id": self.vessel_id[start:end]}