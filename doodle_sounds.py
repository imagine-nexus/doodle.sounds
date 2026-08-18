import os
import sys
import re
import logging
import argparse
import urllib.request
from typing import Tuple

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

# ==========================================
# Application Configuration
# ==========================================
__version__ = "1.0.1"
APP_NAME = "Doodle Sounds Pro"
# Your active GitHub Raw URL
UPDATE_URL = "https://raw.githubusercontent.com/imagine-nexus/doodle.sounds/refs/heads/main/doodle_sounds.py"

# Configure Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(APP_NAME)

# ==========================================
# Auto-Update Mechanism
# ==========================================
def check_for_updates(silent: bool = False) -> None:
    """Checks the remote URL for a newer version, self-updates, and auto-restarts."""
    if not silent:
        logger.info(f"Checking for updates for {APP_NAME} (Current version: {__version__})...")
    
    try:
        req = urllib.request.Request(UPDATE_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            remote_code = response.read().decode('utf-8')
            
        version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', remote_code)
        
        if not version_match:
            if not silent: 
                logger.warning("Could not determine the remote version. Update aborted.")
            return

        remote_version = version_match.group(1)
        
        current_v = tuple(map(int, __version__.split('.')))
        remote_v = tuple(map(int, remote_version.split('.')))

        if remote_v > current_v:
            logger.info(f"New version found: v{remote_version}. Downloading silently...")
            
            # Safely overwrite the current script file
            current_file_path = os.path.abspath(__file__)
            with open(current_file_path, 'w', encoding='utf-8') as f:
                f.write(remote_code)
                
            logger.info("Update installed! Auto-restarting the engine to continue processing...")
            
            # Tell the OS to restart this script with the exact same arguments
            os.execv(sys.executable, ['python', current_file_path] + sys.argv[1:])
            
        else:
            if not silent:
                logger.info("You are running the latest version.")
            
    except Exception as e:
        if not silent:
            logger.error(f"Failed to check for updates: {e}")

# ==========================================
# Core DSP Engine
# ==========================================
class DoodleSoundsEngine:
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    # --- Headphone DSP Effects ---
    def apply_crossfeed(self, left_channel: np.ndarray, right_channel: np.ndarray, 
                        delay_ms: float = 0.3, cutoff_hz: float = 700.0, mix_level: float = -6.0) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates acoustic cross-bleed between ears."""
        if logger.level <= logging.INFO and not hasattr(self, '_crossfeed_logged'):
            logger.info("Applying Acoustic Crossfeed...")
            self._crossfeed_logged = True
            
        delay_samples = int((delay_ms / 1000.0) * self.sr)
        gain = 10 ** (mix_level / 20.0)
        
        nyquist = self.sr / 2.0
        norm_cutoff = cutoff_hz / nyquist
        b, a = signal.butter(1, norm_cutoff, btype='low')
        
        filtered_left = signal.lfilter(b, a, left_channel)
        filtered_right = signal.lfilter(b, a, right_channel)
        
        delayed_left = np.pad(filtered_left, (delay_samples, 0))[:-delay_samples]
        delayed_right = np.pad(filtered_right, (delay_samples, 0))[:-delay_samples]
        
        out_left = left_channel + (delayed_right * gain)
        out_right = right_channel + (delayed_left * gain)
        
        return out_left, out_right

    def add_theater_reverb(self, audio_data: np.ndarray, room_size: float = 0.3) -> np.ndarray:
        """Simulates the acoustic reflections of a large cinema room."""
        logger.info("Simulating IMAX Theater Acoustics...")
        delay_samples = int((40.0 / 1000.0) * self.sr) 
        reverb_tail = np.zeros_like(audio_data)
        reverb_tail[delay_samples:] = audio_data[:-delay_samples] * room_size
        return audio_data + reverb_tail

    def cinematic_bass_boost(self, audio_data: np.ndarray, gain_db: float = 6.0, cutoff_hz: float = 80.0) -> np.ndarray:
        """Mimics a cinema subwoofer by boosting sub-bass frequencies."""
        logger.info(f"Adding Cinematic Subwoofer Boost ({gain_db}dB at {cutoff_hz}Hz)...")
        nyquist = self.sr / 2.0
        w0 = cutoff_hz / nyquist
        A = 10 ** (gain_db / 40.0)
        b, a = signal.iirfilter(2, w0, rs=gain_db, btype='lowpass', analog=False, ftype='butter')
        bass_only = signal.lfilter(b, a, audio_data, axis=0)
        return audio_data + (bass_only * (A - 1.0))

    def cinematic_soft_clipper(self, audio_data: np.ndarray, drive: float = 1.5) -> np.ndarray:
        """Adds analog-style warmth and prevents digital clipping."""
        logger.info("Applying Cinematic Dynamic Saturation...")
        return np.tanh(audio_data * drive)

    # --- Surround Upmixing Matrix ---
    def upmix_to_surround(self, left: np.ndarray, right: np.ndarray, mode: str) -> np.ndarray:
        """Matrix decodes a stereo signal into 5.1 or 7.1 multi-channel audio."""
        logger.info(f"Upmixing stereo matrix to {mode} Surround...")
        
        fl = left * 0.8
        fr = right * 0.8
        center = (left + right) * 0.5
        
        # Subwoofer isolation
        nyquist = self.sr / 2.0
        b_lfe, a_lfe = signal.butter(4, 120.0 / nyquist, btype='low')
        lfe = signal.lfilter(b_lfe, a_lfe, center) * 1.5 
        
        # Surround Ambience isolation
        side_l = left - center
        side_r = right - center
        delay_20ms = int((20.0 / 1000.0) * self.sr)
        sl = np.pad(side_l, (delay_20ms, 0))[:-delay_20ms] * 0.7
        sr = np.pad(side_r, (delay_20ms, 0))[:-delay_20ms] * 0.7
        
        if mode == '5.1':
            return np.column_stack((fl, fr, center, lfe, sl, sr))
            
        elif mode == '7.1':
            delay_40ms = int((40.0 / 1000.0) * self.sr)
            rl = np.pad(side_l, (delay_40ms, 0))[:-delay_40ms] * 0.6
            rr = np.pad(side_r, (delay_40ms, 0))[:-delay_40ms] * 0.6
            return np.column_stack((fl, fr, center, lfe, rl, rr, sl, sr))

# ==========================================
# Utility Functions
# ==========================================
def generate_test_signal(sr: int = 44100, duration: int = 4) -> np.ndarray:
    """Generates a synthetic stereo test signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 40 * t) * np.exp(-t)
    right = 0.5 * np.sin(2 * np.pi * 330 * t) + 0.3 * signal.chirp(t, 200, duration, 1000)
    audio = np.column_stack((left, right)).astype(np.float32)
    return audio / np.max(np.abs(audio))

# ==========================================
# Main Execution Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} - Cinematic DSP Audio Engine")
    parser.add_argument("-i", "--input", type=str, help="Path to input .wav file")
    parser.add_argument("-o", "--output", type=str, help="Path for output .wav file")
    parser.add_argument("-m", "--mode", type=str, choices=['headphone', '5.1', '7.1'], default='headphone', 
                        help="Target output format (default: headphone)")
    parser.add_argument("--update", action="store_true", help="Force a manual update check and exit")
    
    args = parser.parse_args()

    # 1. Handle Explicit Manual Update Request
    if args.update:
        check_for_updates(silent=False)
        sys.exit(0)

    # 2. AUTOMATIC BACKGROUND UPDATE CHECK ON LAUNCH
    check_for_updates(silent=True)

    # 3. Audio Loading and Setup
    sr = 44100
    file_name = args.input or "test_audio.wav"
    
    if os.path.exists(file_name):
        logger.info(f"Loading '{file_name}'...")
        try:
            sr, audio_data = wavfile.read(file_name)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
            if len(audio_data.shape) == 1:
                logger.warning("Audio is mono. Duplicating to stereo...")
                audio_data = np.column_stack((audio_data, audio_data))
            left_ch, right_ch = audio_data[:, 0], audio_data[:, 1]
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            sys.exit(1)
    else:
        logger.info(f"'{file_name}' not found. Generating a synthetic test signal...")
        audio_data = generate_test_signal(sr=sr)
        left_ch, right_ch = audio_data[:, 0], audio_data[:, 1]
        wavfile.write("doodle_original_test.wav", sr, audio_data)

    # 4. Processing Pipeline
    engine = DoodleSoundsEngine(sample_rate=sr)
    logger.info(f"--- Starting Processing Pipeline for {args.mode.upper()} ---")
    
    if args.mode == 'headphone':
        out_l, out_r = engine.apply_crossfeed(left_ch, right_ch)
        processed_audio = np.column_stack((out_l, out_r))
        processed_audio = engine.add_theater_reverb(processed_audio, room_size=0.35)
        processed_audio = engine.cinematic_bass_boost(processed_audio, gain_db=8.0, cutoff_hz=80.0)
    else:
        processed_audio = engine.upmix_to_surround(left_ch, right_ch, args.mode)

    processed_audio = engine.cinematic_soft_clipper(processed_audio, drive=1.4)
    
    # 5. Final Export
    logger.info("Normalizing final audio...")
    max_val = np.max(np.abs(processed_audio))
    if max_val > 0:
        processed_audio /= max_val
        
    output_name = args.output or f"doodle_{args.mode}_mix.wav"
    
    try:
        wavfile.write(output_name, sr, processed_audio.astype(np.float32))
        logger.info(f"SUCCESS! Output saved to: '{output_name}'")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")

if __name__ == "__main__":
    main()