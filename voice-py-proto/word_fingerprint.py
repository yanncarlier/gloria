"""
word_fingerprint.py

Compares multiple audio recordings of a spoken word and determines whether
they are the same word (or same utterance) using:

  1. MFCC feature extraction (robust to loudness / small pitch changes)
  2. DTW (Dynamic Time Warping) distance -> the actual same/different decision,
     tolerant of different speaking rates.
  3. A SimHash-style locality-sensitive hash (LSH) built from the pooled
     MFCC statistics -> a fixed-length bit-string "fingerprint" where
     perceptually similar recordings land close together (small Hamming
     distance), even though the underlying audio differs sample-for-sample.
  4. SHA-256 of that fingerprint -> a fixed-size cryptographic "proof" you
     can store/publish/compare without needing the raw audio. This part is
     NOT fuzzy: a single differing bit in the fingerprint produces a
     completely different SHA-256. Use it for exact re-verification /
     tamper-evidence of a specific fingerprint, and use the fingerprint's
     Hamming distance for fuzzy "is this probably the same word" checks.

Usage:
    python word_fingerprint.py file1.wav file2.wav file3.wav ...

Requires: librosa, numpy, scipy, fastdtw, soundfile
    pip install librosa fastdtw numpy scipy soundfile
"""

import sys
import hashlib
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000          # resample everything to this rate
N_MFCC = 13                  # number of MFCC coefficients
FINGERPRINT_BITS = 128       # length of the SimHash fingerprint, in bits
RANDOM_SEED = 42             # fixed seed -> same hyperplanes every run,
                              # so fingerprints are comparable across runs

# Decision thresholds (tune these for your data / recording conditions).
# DTW distance is normalized by path length, so it's roughly comparable
# across recordings of different durations.
DTW_SAME_WORD_THRESHOLD = 45.0
# Hamming distance out of FINGERPRINT_BITS bits.
HAMMING_SAME_WORD_THRESHOLD = 20


@dataclass
class WordSample:
    path: str
    mfcc_seq: np.ndarray        # (n_frames, N_MFCC) time-series, for DTW
    pooled_vector: np.ndarray   # fixed-length summary vector, for hashing
    fingerprint_bits: np.ndarray  # (FINGERPRINT_BITS,) array of 0/1
    fingerprint_hex: str
    sha256_proof: str


# --------------------------------------------------------------------------
# Step 1: Load audio and extract features
# --------------------------------------------------------------------------

def load_and_trim(path: str) -> np.ndarray:
    """Load audio, resample, and trim leading/trailing silence."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if audio.size == 0:
        raise ValueError(f"'{path}' appears to be silent after trimming.")
    # Normalize amplitude so loudness differences don't dominate.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def extract_mfcc_sequence(audio: np.ndarray) -> np.ndarray:
    """
    Extract an MFCC time-series (n_frames, N_MFCC), used for DTW comparison.
    Delta (velocity) coefficients are appended since they help distinguish
    words with similar timbre but different articulation dynamics.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    combined = np.vstack([mfcc, delta])  # (2*N_MFCC, n_frames)
    return combined.T  # (n_frames, 2*N_MFCC)


def pool_features(mfcc_seq: np.ndarray) -> np.ndarray:
    """
    Collapse the variable-length MFCC time-series into a fixed-length
    summary vector (mean + std per coefficient) so it can be hashed.
    """
    mean = mfcc_seq.mean(axis=0)
    std = mfcc_seq.std(axis=0)
    return np.concatenate([mean, std])


# --------------------------------------------------------------------------
# Step 2: DTW-based same/different decision
# --------------------------------------------------------------------------

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two MFCC sequences, normalized
    by the warping path length so recordings of different durations are
    comparable.
    """
    distance, path = fastdtw(seq_a, seq_b, dist=euclidean)
    return distance / len(path)


# --------------------------------------------------------------------------
# Step 3: SimHash-style fingerprint (locality-sensitive hash)
# --------------------------------------------------------------------------

def _random_hyperplanes(dim: int, n_bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_bits, dim))


def compute_fingerprint(pooled_vector: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    SimHash: project the feature vector onto FINGERPRINT_BITS random
    hyperplanes. Each bit records which side of its hyperplane the vector
    falls on. Vectors that are close in feature space end up with hashes
    that differ in only a few bits (small Hamming distance) -- this is
    what makes the fingerprint tolerant to small acoustic variation.
    """
    hyperplanes = _random_hyperplanes(pooled_vector.shape[0],
                                       FINGERPRINT_BITS, RANDOM_SEED)
    projections = hyperplanes @ pooled_vector
    bits = (projections >= 0).astype(np.uint8)  # 0/1 array

    # Pack bits into hex for a compact, storable string.
    bit_string = "".join(str(b) for b in bits)
    hex_fingerprint = f"{int(bit_string, 2):0{FINGERPRINT_BITS // 4}x}"
    return bits, hex_fingerprint


def hamming_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
    return int(np.sum(bits_a != bits_b))


# --------------------------------------------------------------------------
# Step 4: SHA-256 proof of the fingerprint
# --------------------------------------------------------------------------

def sha256_of_fingerprint(hex_fingerprint: str) -> str:
    """
    A cryptographic commitment to the fingerprint. Two recordings with the
    *identical* fingerprint hex will produce the identical SHA-256 proof;
    this is for exact re-verification / tamper evidence (e.g. "prove this
    fingerprint is the one that was recorded, without exposing the audio"),
    not for fuzzy matching -- use Hamming distance on the fingerprint bits
    for that.
    """
    return hashlib.sha256(hex_fingerprint.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def process_file(path: str) -> WordSample:
    audio = load_and_trim(path)
    mfcc_seq = extract_mfcc_sequence(audio)
    pooled = pool_features(mfcc_seq)
    bits, hex_fp = compute_fingerprint(pooled)
    proof = sha256_of_fingerprint(hex_fp)
    return WordSample(
        path=path,
        mfcc_seq=mfcc_seq,
        pooled_vector=pooled,
        fingerprint_bits=bits,
        fingerprint_hex=hex_fp,
        sha256_proof=proof,
    )


def compare_samples(samples: List[WordSample]) -> None:
    n = len(samples)
    print("\n=== Fingerprints ===")
    for s in samples:
        print(f"{s.path}")
        print(f"  fingerprint (hex, {FINGERPRINT_BITS} bits): {s.fingerprint_hex}")
        print(f"  sha256 proof:                        {s.sha256_proof}")

    print("\n=== Pairwise comparison ===")
    header = f"{'file A':30s} {'file B':30s} {'DTW dist':>10s} {'Hamming':>8s} {'verdict':>12s}"
    print(header)
    print("-" * len(header))

    for i in range(n):
        for j in range(i + 1, n):
            a, b = samples[i], samples[j]
            dtw_d = dtw_distance(a.mfcc_seq, b.mfcc_seq)
            ham_d = hamming_distance(a.fingerprint_bits, b.fingerprint_bits)

            same_by_dtw = dtw_d < DTW_SAME_WORD_THRESHOLD
            same_by_hash = ham_d < HAMMING_SAME_WORD_THRESHOLD
            if same_by_dtw and same_by_hash:
                verdict = "SAME WORD"
            elif same_by_dtw or same_by_hash:
                verdict = "UNCERTAIN"
            else:
                verdict = "DIFFERENT"

            name_a = a.path.split("/")[-1]
            name_b = b.path.split("/")[-1]
            print(f"{name_a:30s} {name_b:30s} {dtw_d:10.2f} {ham_d:8d} {verdict:>12s}")

    print(f"\n(Thresholds: DTW < {DTW_SAME_WORD_THRESHOLD}, "
          f"Hamming < {HAMMING_SAME_WORD_THRESHOLD}/{FINGERPRINT_BITS} bits. "
          f"Tune these for your recording conditions.)")


def main():
    parser = argparse.ArgumentParser(
        description="Fingerprint and compare spoken-word audio files.")
    parser.add_argument("files", nargs="+", help="Paths to audio files (wav/mp3/etc).")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Provide at least 2 files to compare.", file=sys.stderr)
        sys.exit(1)

    samples = [process_file(f) for f in args.files]
    compare_samples(samples)


if __name__ == "__main__":
    main()
