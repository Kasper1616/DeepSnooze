from pathlib import Path
import torch

class SleepyRatCachedDataset(torch.utils.data.Dataset):
    def __init__(self, processed_path="data_processed/"):
        self.processed_path = Path(processed_path)
        self.files = sorted(list(self.processed_path.glob("*.pt")))
        
        self.index_map = []
        
        print("Initializing Cached Dataset...")
        # We need to know which file and which index corresponds to global index i
        # This loop is extremely fast compared to opening EDFs
        for file_idx, f in enumerate(self.files):
            # We load just the labels to get the length (lightweight)
            # efficient way: map_location='cpu' prevents GPU overloading
            data_dict = torch.load(f, map_location='cpu', weights_only=True)
            num_samples = len(data_dict["y"])
            
            for i in range(num_samples):
                self.index_map.append((file_idx, i))
                
        # Optional: Cache the data in RAM if you have enough memory (e.g. <16GB total data)
        # self.cache = [torch.load(f) for f in self.files] 

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        
        # Option A: Low RAM usage (Load file on demand)
        # This is fast because .pt files are optimized for binary reading
        file_path = self.files[file_idx]
        data_dict = torch.load(file_path, weights_only=True)
        
        signal = data_dict["X"][sample_idx] # [3, 512]
        label = data_dict["y"][sample_idx]  # scalar
        

        return signal, label

if __name__ == "__main__":
    dataset = SleepyRatCachedDataset()
    print(f"Total samples in cached dataset: {len(dataset)}")
    signal, label = dataset[0]
    print(f"Sample signal shape: {signal.shape}, label: {label}")