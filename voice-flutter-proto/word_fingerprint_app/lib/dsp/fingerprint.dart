// DTW same/different comparison, SimHash-style fuzzy fingerprint, and a
// SHA-256 "proof" of that fingerprint. Mirrors the Python reference
// implementation's design.

import 'dart:convert';
import 'dart:math' as math;
import 'package:crypto/crypto.dart';

const int fingerprintBits = 128;
const int _randomSeed = 42;

/// Collapses a variable-length MFCC time series into a fixed-length
/// summary vector (per-coefficient mean + std) so it can be hashed.
List<double> poolFeatures(List<List<double>> mfccSeq) {
  final nFrames = mfccSeq.length;
  final nCoeffs = mfccSeq[0].length;

  final mean = List<double>.filled(nCoeffs, 0);
  for (final frame in mfccSeq) {
    for (int c = 0; c < nCoeffs; c++) {
      mean[c] += frame[c];
    }
  }
  for (int c = 0; c < nCoeffs; c++) {
    mean[c] /= nFrames;
  }

  final variance = List<double>.filled(nCoeffs, 0);
  for (final frame in mfccSeq) {
    for (int c = 0; c < nCoeffs; c++) {
      final diff = frame[c] - mean[c];
      variance[c] += diff * diff;
    }
  }
  final std = List<double>.filled(nCoeffs, 0);
  for (int c = 0; c < nCoeffs; c++) {
    std[c] = math.sqrt(variance[c] / nFrames);
  }

  return [...mean, ...std];
}

/// Dynamic Time Warping distance between two MFCC sequences, normalized
/// by path length so recordings of different durations are comparable.
double dtwDistance(List<List<double>> a, List<List<double>> b) {
  final n = a.length;
  final m = b.length;
  final dist =
      List.generate(n + 1, (_) => List<double>.filled(m + 1, double.infinity));
  dist[0][0] = 0;

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      final cost = _euclidean(a[i - 1], b[j - 1]);
      final minPrev = math.min(dist[i - 1][j], math.min(dist[i][j - 1], dist[i - 1][j - 1]));
      dist[i][j] = cost + minPrev;
    }
  }

  return dist[n][m] / (n + m);
}

double _euclidean(List<double> x, List<double> y) {
  double sum = 0;
  for (int i = 0; i < x.length; i++) {
    final d = x[i] - y[i];
    sum += d * d;
  }
  return math.sqrt(sum);
}

class Fingerprint {
  final List<int> bits; // 0/1, length == fingerprintBits
  final String hex;
  final String sha256Proof;

  Fingerprint(this.bits, this.hex, this.sha256Proof);
}

List<List<double>>? _cachedHyperplanes;
int _cachedDim = -1;

/// SimHash: project the pooled feature vector onto `fingerprintBits` fixed
/// random hyperplanes. Vectors close in feature space land within a small
/// Hamming distance of each other -- this is what makes the fingerprint
/// tolerant to small acoustic variation between takes.
List<List<double>> _hyperplanes(int dim) {
  if (_cachedHyperplanes != null && _cachedDim == dim) return _cachedHyperplanes!;
  final rng = math.Random(_randomSeed);
  final planes = List.generate(
    fingerprintBits,
    (_) => List.generate(dim, (_) => _standardNormal(rng)),
  );
  _cachedHyperplanes = planes;
  _cachedDim = dim;
  return planes;
}

/// Box-Muller transform: converts uniform randoms into a standard normal
/// sample, matching the distribution numpy's default_rng().normal() draws.
double _standardNormal(math.Random rng) {
  final u1 = 1 - rng.nextDouble(); // avoid log(0)
  final u2 = rng.nextDouble();
  return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
}

Fingerprint computeFingerprint(List<double> pooledVector) {
  final planes = _hyperplanes(pooledVector.length);
  final bits = List<int>.filled(fingerprintBits, 0);

  for (int i = 0; i < fingerprintBits; i++) {
    double proj = 0;
    for (int d = 0; d < pooledVector.length; d++) {
      proj += planes[i][d] * pooledVector[d];
    }
    bits[i] = proj >= 0 ? 1 : 0;
  }

  final bitString = bits.join();
  final hex = BigInt.parse(bitString, radix: 2)
      .toRadixString(16)
      .padLeft(fingerprintBits ~/ 4, '0');

  // Cryptographic commitment to the fingerprint. Identical fingerprints
  // produce identical proofs; a single differing bit changes it entirely,
  // so use Hamming distance (not this proof) for fuzzy matching.
  final proof = sha256.convert(utf8.encode(hex)).toString();

  return Fingerprint(bits, hex, proof);
}

int hammingDistance(List<int> a, List<int> b) {
  int count = 0;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) count++;
  }
  return count;
}
