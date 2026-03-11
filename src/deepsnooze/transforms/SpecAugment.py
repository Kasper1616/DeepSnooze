import torchaudio.transforms as T
import torchvision.transforms as transforms
import torch

class SpecAugment:
    def __init__(self, freq_mask_param=5, time_mask_param=4):
        # freq_mask_param: Max number of consecutive frequency bins to mask
        # time_mask_param: Max number of consecutive time frames to mask
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def __call__(self, spec):
        """
        Args:
            spec (Tensor): Normalized log-spectrogram from your SpectrogramTransform
                           Shape: (3, Freq, Time)
        """
        # Apply frequency masking
        augmented = self.freq_mask(spec)
        # Apply time masking
        augmented = self.time_mask(augmented)
        
        return augmented