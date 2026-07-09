"""
make_test_audio.py

Generates synthetic .wav files to sanity-check word_fingerprint.py without
needing real recordings. Creates:
  - word_a_take1.wav, word_a_take2.wav, word_a_take3.wav: three slightly
    different renderings of the "same word" (same formant-like frequency
    structure, small random timing/pitch/noise jitter per take -- mimicking
    natural repetition variance).
  - word_b_take1.wav: a clearly different "word" (different frequency
    structure), to confirm the pipeline tells them apart.

Requires: numpy, soundfile
    pip install numpy soundfile
"""

import numpy as np
import soundfile as sf

SR = 16000


def synth_word(freqs, amps, duration=0.6, jitter_seed=0, noise_level=0.01):
    """
    Build a toy 'word' as a sum of a few sinusoids with an amplitude
    envelope, standing in for formant structure. Adding a per-take random
    jitter to duration/pitch/noise simulates natural recording variance.
    """
    rng = np.random.default_rng(jitter_seed)
    dur = duration * (1 + rng.uniform(-0.15, 0.15))  # timing jitter
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)

    signal = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        f_jittered = f * (1 + rng.uniform(-0.02, 0.02))  # pitch jitter
        signal += a * np.sin(2 * np.pi * f_jittered * t)

    # Simple envelope: fade in/out so it doesn't click, roughly word-shaped.
    envelope = np.ones_like(t)
    fade = int(0.1 * SR * dur / duration)
    fade = max(fade, 1)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope

    signal += rng.normal(scale=noise_level, size=signal.shape)  # mic noise
    signal /= np.max(np.abs(signal))
    return signal.astype(np.float32)


def main():
    # "word A": formant-like stack at ~300/900/2200 Hz, three takes.
    freqs_a = [300, 900, 2200]
    amps_a = [1.0, 0.6, 0.3]
    for i, seed in enumerate([1, 2, 3], start=1):
        audio = synth_word(freqs_a, amps_a, jitter_seed=seed)
        sf.write(f"/home/claude/word_a_take{i}.wav", audio, SR)

    # "word B": clearly different formant structure -> different word.
    freqs_b = [500, 1500, 3000]
    amps_b = [1.0, 0.5, 0.4]
    audio_b = synth_word(freqs_b, amps_b, jitter_seed=99)
    sf.write("/home/claude/word_b_take1.wav", audio_b, SR)

    print("Wrote word_a_take1.wav, word_a_take2.wav, word_a_take3.wav, word_b_take1.wav")


if __name__ == "__main__":
    main()
