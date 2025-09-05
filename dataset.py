import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any


class SequentialDataset(Dataset):
    """
    PyTorch Dataset for sequential time series data with sliding windows.
    
    This dataset loads data based on JSON files that contain window information
    (filename and start_idx) and extracts sequential windows for training/testing.
    """
    
    def __init__(
        self, 
        json_file: str,
        window_size: int = 18,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            json_file: Path to JSON file containing window information
            window_size: Size of each sequential window
            feature_columns: List of feature column names. If None, all except target columns
        """
        self.window_size = window_size
        # Load window information from JSON
        with open(json_file, 'r') as f:
            self.data_file = json.load(f)
        
        # Cache for loaded dataframes to avoid repeated file reads
        self._dataframe_cache = {}
        
        # Load first dataframe to determine columns
        first_window = self.data_file[0]
        sample_df = self._load_dataframe(first_window['filename'])
        
        # Set feature and target columns

        self.target_columns = ['1_sampling', '2_sampling', '3_sampling', 
                                 '4_sampling', '5_sampling', '6_sampling']
    
        if feature_columns is None:
            # Use all columns except target columns and time
            all_columns = list(sample_df.columns)
            excluded = self.target_columns + ['time']
            self.feature_columns = [col for col in all_columns if col not in excluded]
        else:
            self.feature_columns = feature_columns

        print(f"Dataset initialized with {len(self.data_file)} windows")
        print(f"Feature columns ({len(self.feature_columns) + 6}): {self.feature_columns[:5]}...")
        print(f"Target columns ({len(self.target_columns)}): {self.target_columns}")
    
    def _load_dataframe(self, filename: str) -> pd.DataFrame:
        """Load dataframe with caching."""
        if filename not in self._dataframe_cache:
            self._dataframe_cache[filename] = pd.read_csv(filename)
        return self._dataframe_cache[filename]
    
    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        return len(self.data_file)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single window of sequential data.
        
        Args:
            idx: Index of the window
            
        Returns:
            Tuple of (features, targets) as torch tensors
            - features: (window_size, num_features)
            - targets: (window_size, num_targets)
        """
        window = self.data_file[idx]
        df = self._load_dataframe(window['filename'])
        
        start_idx = window['start_idx']
        end_idx = start_idx + self.window_size
        
        # Extract window data
        
        window_data = df.iloc[start_idx:end_idx]
        
        # Extract features and targets
        features = window_data[self.feature_columns].values.astype(np.float32)
        labels = window_data['label'].values.astype(np.int64)
        # Convert labels (1-6) to one-hot encoding (6 classes)
        labels_onehot = np.eye(6)[labels - 1].astype(np.float32)  # Subtract 1 to make it 0-5 indexed
        features = np.concatenate([features, labels_onehot], axis=-1)
        targets = window_data[self.target_columns].values.astype(np.float32)

        return torch.tensor(features), torch.tensor(targets)



# Example usage and testing
if __name__ == "__main__":
    # Example usage
    root_path = "processed_data"
    window_size = 18
    
    # File paths
    train_json = f"{root_path}/train_data_windows_{window_size}.json"
    test_json = f"{root_path}/test_data_windows_{window_size}.json"
    metadata_file = f"{root_path}/dataset_metadata_{window_size}.json"
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [train_json, test_json, metadata_file]):
        print("JSON files not found. Please run the data preprocessing notebook first.")
        exit(1)
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"Metadata loaded: {metadata}")

    dataset_train = SequentialDataset(train_json, window_size=window_size, full_window_step=2)
    dataset_test = SequentialDataset(test_json, window_size=window_size, full_window_step=2)

    data = dataset_train.__getitem__(0)
    print(f"First training sample features shape: {data[0].shape}")
    print(data[0][-1])
    print(f"First training sample targets shape: {data[1].shape}")
    print(data[1][-1])