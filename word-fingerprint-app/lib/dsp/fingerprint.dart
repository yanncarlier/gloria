// DTW same/different comparison, richer pooled MFCC summaries, and a
// SHA-256 "proof" of that fingerprint. Mirrors the Python reference
// implementation's design.

import 'dart:convert';
import 'dart:math' as math;
import 'package:crypto/crypto.dart';

const int fingerprintBits = 128;
const int _randomSeed = 42;

class VoiceComparisonResult {
  final double dtwDistance;
  final int hammingDistance;
  final double featureDistance;
  final bool isSameWord;
  final String verdict;

  const VoiceComparisonResult({
    required this.dtwDistance,
    required this.hammingDistance,
    required this.featureDistance,
    required this.isSameWord,
    required this.verdict,
  });
}

/// Collapses a variable-length MFCC time series into a fixed-length
/// summary vector (mean + std + delta mean + delta std) so it can be hashed.
List<double> poolFeatures(List<List<double>> mfccSeq) {
  if (mfccSeq.isEmpty) return <double>[];
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

  final deltaMean = List<double>.filled(nCoeffs, 0);
  final deltaStd = List<double>.filled(nCoeffs, 0);
  for (int c = 0; c < nCoeffs; c++) {
    final deltas = <double>[];
    for (int t = 1; t < nFrames; t++) {
      deltas.add(mfccSeq[t][c] - mfccSeq[t - 1][c]);
    }
    if (deltas.isEmpty) {
      deltaMean[c] = 0;
      deltaStd[c] = 0;
      continue;
    }

    double deltaSum = 0;
    for (final value in deltas) {
      deltaSum += value;
    }
    final deltaAvg = deltaSum / deltas.length;
    double deltaVariance = 0;
    for (final value in deltas) {
      final diff = value - deltaAvg;
      deltaVariance += diff * diff;
    }
    deltaMean[c] = deltaAvg;
    deltaStd[c] = math.sqrt(deltaVariance / deltas.length);
  }

  final energyValues = <double>[];
  for (final frame in mfccSeq) {
    double energy = 0;
    for (final coeff in frame) {
      energy += coeff.abs();
    }
    energyValues.add(energy);
  }

  final energyMean = energyValues.isEmpty
      ? 0.0
      : energyValues.reduce((a, b) => a + b) / energyValues.length;
  double energyVariance = 0;
  for (final value in energyValues) {
    final diff = value - energyMean;
    energyVariance += diff * diff;
  }
  final energyStd = energyValues.isEmpty ? 0.0 : math.sqrt(energyVariance / energyValues.length);

  final features = <double>[...mean, ...std, ...deltaMean, ...deltaStd, energyMean, energyStd];
  return _normalizeVector(features);
}

List<double> _normalizeVector(List<double> values) {
  if (values.isEmpty) return values;
  double maxAbs = 0;
  for (final value in values) {
    final absValue = value.abs();
    if (absValue > maxAbs) maxAbs = absValue;
  }
  if (maxAbs < 1e-12) return values;
  return values.map((value) => value / maxAbs).toList();
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
      final minPrev = math.min(
        dist[i - 1][j],
        math.min(dist[i][j - 1], dist[i - 1][j - 1]),
      );
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

/// Euclidean distance between two pooled feature vectors; smaller values mean
/// the recordings are structurally more similar.
double featureDistance(List<double> a, List<double> b) {
  if (a.isEmpty || b.isEmpty || a.length != b.length) return 1.0;

  double sum = 0;
  for (int i = 0; i < a.length; i++) {
    final diff = a[i] - b[i];
    sum += diff * diff;
  }

  return math.sqrt(sum).clamp(0.0, 10.0);
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

VoiceComparisonResult compareVoiceSequences(
  List<List<double>> a,
  List<List<double>> b, {
  double dtwSameWordThreshold = 1.5,
  int hammingSameWordThreshold = 10,
  double featureSameWordThreshold = 0.2,
}) {
  final pooledA = poolFeatures(a);
  final pooledB = poolFeatures(b);
  final fingerprintA = computeFingerprint(pooledA);
  final fingerprintB = computeFingerprint(pooledB);

  final dtw = dtwDistance(a, b);
  final hamming = hammingDistance(fingerprintA.bits, fingerprintB.bits);
  final feature = featureDistance(pooledA, pooledB);

  final sameByDtw = dtw < dtwSameWordThreshold;
  final sameByHash = hamming < hammingSameWordThreshold;
  final sameByFeature = feature < featureSameWordThreshold;
  final isSameWord = sameByDtw && sameByHash && sameByFeature;
  final verdict = isSameWord ? 'SAME WORD' : 'DIFFERENT';

  return VoiceComparisonResult(
    dtwDistance: dtw,
    hammingDistance: hamming,
    featureDistance: feature,
    isSameWord: isSameWord,
    verdict: verdict,
  );
}
