import os
import json
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader


class MyHiddenStatesDataset(Dataset):
    """
    A custom Dataset that looks for directories under `root_dir`, each containing
    'hidden_state_layer_x.npy' and 'data.json'.

    Args:
        root_dir (str): Path to the dataset root directory.
        hidden_state_filename (str): The filename for the hidden states `.npy`.
                                     Defaults to 'hidden_state_layer_x.npy'.
    """

    def __init__(self, root_dir, hidden_state_filename_long):
        super().__init__()
        self.root_dir = root_dir
        self.hidden_state_filename_long = hidden_state_filename_long

        # Each subfolder is treated as one sample
        self.sample_dirs = sorted(
            d for d in glob(os.path.join(root_dir, '*')) if os.path.isdir(d)
        )

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        data_dir = self.sample_dirs[idx]

        hidden_states_long, meta_long = self.load_hidden_states_and_meta(
            data_dir, self.hidden_state_filename_long, 'long'
        )

        # meta might look like {"needle_idx_range": [25, 174], ...}
        # We return (hidden_states, meta)
        return hidden_states_long, meta_long

    def load_hidden_states_and_meta(self, data_dir, hidden_state_filename, data_json_name):
        # 1) Load the hidden states
        hs_path = os.path.join(data_dir, hidden_state_filename)
        hidden_states_np = np.load(hs_path)  # shape [bsz, seq_len, hidden_size] or similar
        hidden_states = torch.from_numpy(hidden_states_np)
        # 2) Load the JSON metadata
        json_path = os.path.join(data_dir, f'{data_json_name}.json')
        with open(json_path, 'r') as f:
            meta = json.load(f)
        return hidden_states, meta
