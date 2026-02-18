import torch
import torchaudio.transforms as T

class SpectrogramTransform:
    def __init__(self, n_fft=64, hop_length=32):
        
        self.spec_layer = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,           # 2.0 = Power Spectrogram (Intensity)
            normalized=True      # Normalizes by window size
        )

    def __call__(self, x):
        """
        Args:
            x (Tensor): Input signal of shape (Channels, Time) -> (3, 512)
        Returns:
            Tensor: Log-scaled spectrogram of shape (3, Freq, Time)
        """
        spec = self.spec_layer(x)

        log_spec = torch.log(spec + 1e-6)

        mean = log_spec.mean(dim=(1, 2), keepdim=True)
        std = log_spec.std(dim=(1, 2), keepdim=True)
        
        return (log_spec - mean) / (std + 1e-6)