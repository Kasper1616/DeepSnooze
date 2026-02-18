import torch
import torchaudio.transforms as T

# --- YOUR CONFIG ---
FS = 128            # Your sampling rate
WINDOW_SIZE = 512   # Your input size (4 seconds)
N_FFT = 64          # Frequency resolution
HOP_LENGTH = 32     # Time resolution
# -------------------

def check_spectrogram_physics():
    dummy_signal = torch.randn(1, WINDOW_SIZE)
    
    transform = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    spec = transform(dummy_signal)
    
    freq_bins = spec.shape[1]
    time_steps = spec.shape[2]
    
    bin_width = FS / N_FFT
    total_time_step_duration = HOP_LENGTH / FS
    
    print(f"--- SPECTROGRAM PHYSICS CHECK ---")
    print(f"Input Signal: {WINDOW_SIZE} samples ({WINDOW_SIZE/FS:.2f} seconds)")
    print(f"Output Image Shape: {spec.shape[1]} (Freq) x {spec.shape[2]} (Time)")
    print(f"\nVERTICAL AXIS (Frequency):")
    print(f"  - Resolution: Each pixel is {bin_width} Hz tall")
    print(f"  - Max Freq: {freq_bins * bin_width} Hz (Nyquist Limit)")
    print(f"  - Can we see Theta (6-9Hz)? {'YES' if bin_width <= 2 else 'NO (Too blurry)'}")
    
    print(f"\nHORIZONTAL AXIS (Time):")
    print(f"  - Resolution: Each pixel is {total_time_step_duration:.3f} seconds wide")
    print(f"---------------------------------")

if __name__ == "__main__":
    check_spectrogram_physics()