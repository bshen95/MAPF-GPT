import glob
import os
import time

import numpy as np
import pyarrow as pa
import torch

from loguru import logger
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import zip_longest
from typing import List

class MapfArrowDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, device, batch_size):
        self.all_data_files = self.file_paths = sorted(glob.glob(os.path.join(folder_path, "*.arrow")))
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.int8

        ddp_local_rank = os.environ.get("LOCAL_RANK")
        ddp_world_size = os.environ.get("WORLD_SIZE")
        # Divide files among DDP workers for training
        if "train" in folder_path and ddp_local_rank is not None and ddp_world_size is not None:
            ddp_local_rank, ddp_world_size = int(ddp_local_rank), int(ddp_world_size)
            files_per_worker = len(self.file_paths) // ddp_world_size
            start_index = ddp_local_rank * files_per_worker
            end_index = start_index + files_per_worker
            self.file_paths = self.file_paths[start_index:end_index]

        # pre-allocate memory for the input and target tensors (same file size)
        sample_input_tensors, sample_gt_actions = self._get_data_from_file(self.file_paths[0])

        self.input_tensors = torch.empty(sample_input_tensors.shape, dtype=self.dtype, device=self.device)
        self.target_tensors = torch.full(sample_input_tensors.shape, -1, dtype=self.dtype, device=self.device)

        logger.info(f"Single file tensor size: {self.input_tensors.numel() * self.input_tensors.element_size() / 1e9:.4f} GB")

    @staticmethod
    def _get_data_from_file(file_path):
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy(zero_copy_only=False)
            gt_actions = table["gt_actions"].to_numpy(zero_copy_only=False)

        # shuffle data within the current file
        indices = np.random.permutation(len(input_tensors))
        input_tensors = np.stack(input_tensors[indices])
        gt_actions = gt_actions[indices]

        return input_tensors, gt_actions

    def load_and_transfer_data_file(self, filename):
        start_time = time.monotonic()

        input_tensors_np, gt_actions_np = self._get_data_from_file(filename)

        input_tensor_torch = torch.tensor(input_tensors_np, dtype=self.dtype, device=self.device)
        gt_actions_torch = torch.tensor(gt_actions_np, dtype=self.dtype, device=self.device)

        # Resize internal buffers if needed
        if input_tensor_torch.shape != self.input_tensors.shape:
            logger.warning(f"Resizing buffers: from {self.input_tensors.shape} to {input_tensor_torch.shape}")
            self.input_tensors = torch.empty_like(input_tensor_torch, device=self.device)
            self.target_tensors = torch.full_like(input_tensor_torch, -1, device=self.device)
            

        # self.input_tensors.copy_(torch.tensor(input_tensors, dtype=self.dtype), non_blocking=True)
        # self.target_tensors[:, -1].copy_(torch.tensor(gt_actions, dtype=self.dtype), non_blocking=True)
        self.input_tensors.copy_(input_tensor_torch, non_blocking=True)
        self.target_tensors[:, -1].copy_(gt_actions_torch, non_blocking=True)

        finish_time = time.monotonic() - start_time
        logger.debug(f'Data from {filename} for {self.device} device prepared in ~{round(finish_time, 5)}s')

    def interleave_by_map_type(self, file_lists: List[str]) -> List[str]:
        """
        Interleaves dataset file paths by environment/map type
        so that each type is processed in turns.

        Args:
            file_lists (List[str]): List of .arrow file paths.

        Returns:
            List[str]: Interleaved file list.
        """
        # Group by type keywords
        env_groups = defaultdict(list)
        for f in file_lists:
            if 'empty' in f:
                env_groups['empty'].append(f)
            elif 'maze' in f:
                env_groups['maze'].append(f)
            elif 'random' in f:
                env_groups['random'].append(f)
            elif 'room' in f:
                env_groups['room'].append(f)

        # Interleave the grouped files
        interleaved = []
        for group in zip_longest(
            env_groups['empty'],
            env_groups['maze'],
            env_groups['random'],
            env_groups['room']
        ):
            interleaved.extend([x for x in group if x is not None])

        return interleaved

    def __iter__(self):
        while True:
            self.file_paths = self.interleave_by_map_type(self.file_paths)
            for file_path in self.file_paths:
                logger.debug(f'Loading data from {file_path} for {self.device} device')
                self.load_and_transfer_data_file(file_path)
                num_samples = self.input_tensors.shape[0]
                for i in range(0, num_samples, self.batch_size):
                    if i + self.batch_size > num_samples:
                        continue  # Drop last incomplete batch
                    yield self.input_tensors[i:i + self.batch_size], self.target_tensors[i:i + self.batch_size]

                # for i in range(0, len(self.input_tensors), self.batch_size):
                #     if i + self.batch_size > num_samples:
                #         continue  # Drop last incomplete batch
                #     yield self.input_tensors[i:i + self.batch_size], self.target_tensors[i:i + self.batch_size]

    def get_shard_size(self):
        return len(self.input_tensors) * len(self.file_paths)

    def get_full_dataset_size(self):
        return len(self.input_tensors) * len(self.all_data_files)


def main():
    # folder_path = "../dataset/validation"
    folder_path = "../dataset/train"
    dataset = MapfArrowDataset(folder_path, device='cuda:0', batch_size=32)
    data = iter(dataset)
    x = 0
    logger.info(dataset.get_full_dataset_size())
    logger.info(dataset.get_shard_size())

    while True:
        x += 1
        qx, qy = next(data)
        # logger.info(str(qx.shape) + ' ' + str(qy.shape))


if __name__ == "__main__":
    main()

