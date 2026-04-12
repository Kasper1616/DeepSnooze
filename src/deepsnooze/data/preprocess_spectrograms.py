import os
import torch
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm

class SpectrogramProcessor:
    def __init__(self, n_fft=64, hop_length=32):
        self.spec_layer = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            normalized=True
        )

    def process(self, x):
        spec = self.spec_layer(x)
        log_spec = torch.log(spec + 1e-6)
        mean = log_spec.mean(dim=(1, 2), keepdim=True)
        std = log_spec.std(dim=(1, 2), keepdim=True)
        return (log_spec - mean) / (std + 1e-6)

def preprocess_dataset(input_dir, output_dir, n_fft=64, hop_length=32):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processor = SpectrogramProcessor(n_fft=n_fft, hop_length=hop_length)
    files = list(input_path.glob("*.pt"))

    for file_path in tqdm(files):
        data = torch.load(file_path, weights_only=False)
        
        if isinstance(data, dict):
            signal = data["X"]
            label = data["y"]
        else:
            signal = data
            label = None

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
            
        spectrogram = processor.process(signal)
        
        output_data = {"X": spectrogram, "y": label} if label is not None else spectrogram
        torch.save(output_data, output_path / file_path.name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output)