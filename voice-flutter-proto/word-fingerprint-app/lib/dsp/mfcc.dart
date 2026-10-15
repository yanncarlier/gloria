// Pure-Dart MFCC extraction: pre-emphasis -> framing -> Hamming window ->
// FFT magnitude spectrum -> mel filterbank -> log -> DCT-II.
// Mirrors the standard librosa-style MFCC pipeline closely enough for
// same/different-word comparison purposes.

import 'dart:math' as math;
import 'fft.dart';

class MfccExtractor {
  final int frameLengthMs;
  final int hopLengthMs;
  final int numMelFilters;
  final int numCoeffs;
  final double preemphasisCoeff;

  int? _cachedFftSize;
  int? _cachedSampleRate;
  List<List<double>>? _cachedMelFilterbank;

  MfccExtractor({
    this.frameLengthMs = 25,
    this.hopLengthMs = 10,
    this.numMelFilters = 26,
    this.numCoeffs = 13,
    this.preemphasisCoeff = 0.97,
  });

  /// Returns a (n_frames x numCoeffs) MFCC time series for [rawSamples].
  List<List<double>> extract(List<double> rawSamples, int sampleRate) {
    final trimmed = _trimSilence(rawSamples);
    final normalized = _peakNormalize(trimmed);
    final emphasized = _preemphasis(normalized);

    final frameLength = (frameLengthMs * sampleRate / 1000).round();
    final hopLength = (hopLengthMs * sampleRate / 1000).round();
    final fftSize = _nextPowerOfTwo(frameLength);

    final melFilterbank = _getMelFilterbank(sampleRate, fftSize);

    final frames = <List<double>>[];
    for (int start = 0;
        start + frameLength <= emphasized.length;
        start += hopLength) {
      frames.add(emphasized.sublist(start, start + frameLength));
    }
    // Guarantee at least one frame even for very short clips.
    if (frames.isEmpty && emphasized.isNotEmpty) {
      final padded = List<double>.from(emphasized);
      while (padded.length < frameLength) {
        padded.add(0);
      }
      frames.add(padded.sublist(0, frameLength));
    }
    if (frames.isEmpty) {
      throw StateError('Recording contained no usable audio after trimming.');
    }

    final mfccSequence = <List<double>>[];
    for (final frame in frames) {
      final windowed = _applyHammingWindow(frame);
      final spectrum = _magnitudeSpectrum(windowed, fftSize);
      final melEnergies = _applyMelFilterbank(spectrum, melFilterbank);
      final logEnergies =
          melEnergies.map((e) => math.log(math.max(e, 1e-10))).toList();
      final coeffs = _dct(logEnergies, numCoeffs);
      mfccSequence.add(coeffs);
    }

    return normalizeMfccSequence(mfccSequence);
  }

  List<List<double>> normalizeMfccSequence(List<List<double>> mfccSequence) {
    if (mfccSequence.isEmpty) return mfccSequence;

    final nFrames = mfccSequence.length;
    final nCoeffs = mfccSequence[0].length;

    final means = List<double>.filled(nCoeffs, 0);
    for (final frame in mfccSequence) {
      for (int c = 0; c < nCoeffs; c++) {
        means[c] += frame[c];
      }
    }
    for (int c = 0; c < nCoeffs; c++) {
      means[c] /= nFrames;
    }

    final centered = List<List<double>>.generate(
      nFrames,
      (frameIndex) => List<double>.generate(nCoeffs, (coeffIndex) {
        return mfccSequence[frameIndex][coeffIndex] - means[coeffIndex];
      }),
    );

    final variances = List<double>.filled(nCoeffs, 0);
    for (final frame in centered) {
      for (int c = 0; c < nCoeffs; c++) {
        final diff = frame[c];
        variances[c] += diff * diff;
      }
    }
    final stds = List<double>.filled(nCoeffs, 0);
    for (int c = 0; c < nCoeffs; c++) {
      stds[c] = math.sqrt(variances[c] / nFrames);
    }

    return List<List<double>>.generate(nFrames, (frameIndex) {
      return List<double>.generate(nCoeffs, (coeffIndex) {
        final std = stds[coeffIndex];
        return std < 1e-12 ? 0.0 : centered[frameIndex][coeffIndex] / std;
      });
    });
  }

  List<double> _trimSilence(List<double> samples, {double thresholdRatio = 0.02}) {
    if (samples.isEmpty) return samples;
    double maxAbs = 0;
    for (final s in samples) {
      final a = s.abs();
      if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs == 0) return samples;
    final threshold = maxAbs * thresholdRatio;

    int start = 0;
    while (start < samples.length && samples[start].abs() < threshold) {
      start++;
    }
    int end = samples.length - 1;
    while (end > start && samples[end].abs() < threshold) {
      end--;
    }
    if (start >= end) return samples;
    return samples.sublist(start, end + 1);
  }

  List<double> _peakNormalize(List<double> samples) {
    if (samples.isEmpty) return samples;
    double maxAbs = 0;
    for (final s in samples) {
      final a = s.abs();
      if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs == 0) return samples;
    return samples.map((s) => s / maxAbs).toList();
  }

  List<double> _preemphasis(List<double> samples) {
    if (samples.isEmpty) return samples;
    final out = List<double>.filled(samples.length, 0);
    out[0] = samples[0];
    for (int i = 1; i < samples.length; i++) {
      out[i] = samples[i] - preemphasisCoeff * samples[i - 1];
    }
    return out;
  }

  List<double> _applyHammingWindow(List<double> frame) {
    final n = frame.length;
    final out = List<double>.filled(n, 0);
    for (int i = 0; i < n; i++) {
      final w = 0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1));
      out[i] = frame[i] * w;
    }
    return out;
  }

  List<double> _magnitudeSpectrum(List<double> windowedFrame, int fftSize) {
    final complexInput = List<Complex>.generate(fftSize, (i) {
      if (i < windowedFrame.length) return Complex(windowedFrame[i], 0);
      return const Complex(0, 0);
    });
    final result = fft(complexInput);
    final half = fftSize ~/ 2 + 1;
    return List<double>.generate(half, (i) => result[i].magnitude);
  }

  List<List<double>> _getMelFilterbank(int sampleRate, int fftSize) {
    if (_cachedMelFilterbank != null &&
        _cachedFftSize == fftSize &&
        _cachedSampleRate == sampleRate) {
      return _cachedMelFilterbank!;
    }

    double hzToMel(double hz) => 2595 * (math.log(1 + hz / 700) / math.ln10);
    double melToHz(double mel) => 700 * (math.pow(10, mel / 2595) - 1);

    final melLow = hzToMel(0);
    final melHigh = hzToMel(sampleRate / 2);
    final melPoints = List<double>.generate(
      numMelFilters + 2,
      (i) => melLow + (melHigh - melLow) * i / (numMelFilters + 1),
    );
    final hzPoints = melPoints.map(melToHz).toList();
    final binPoints = hzPoints
        .map((hz) => ((fftSize + 1) * hz / sampleRate).floor())
        .toList();

    final numBins = fftSize ~/ 2 + 1;
    final filterbank =
        List.generate(numMelFilters, (_) => List<double>.filled(numBins, 0));

    for (int m = 1; m <= numMelFilters; m++) {
      final left = binPoints[m - 1];
      final center = binPoints[m];
      final right = binPoints[m + 1];

      if (center > left) {
        for (int k = left; k < center; k++) {
          if (k < 0 || k >= numBins) continue;
          filterbank[m - 1][k] = (k - left) / (center - left);
        }
      }
      if (right > center) {
        for (int k = center; k < right; k++) {
          if (k < 0 || k >= numBins) continue;
          filterbank[m - 1][k] = (right - k) / (right - center);
        }
      }
    }

    _cachedMelFilterbank = filterbank;
    _cachedFftSize = fftSize;
    _cachedSampleRate = sampleRate;
    return filterbank;
  }

  List<double> _applyMelFilterbank(
      List<double> spectrum, List<List<double>> filterbank) {
    final energies = List<double>.filled(filterbank.length, 0);
    for (int m = 0; m < filterbank.length; m++) {
      double sum = 0;
      for (int k = 0; k < spectrum.length; k++) {
        sum += (spectrum[k] * spectrum[k]) * filterbank[m][k];
      }
      energies[m] = sum;
    }
    return energies;
  }

  List<double> _dct(List<double> input, int numCoeffs) {
    final n = input.length;
    final output = List<double>.filled(numCoeffs, 0);
    for (int k = 0; k < numCoeffs; k++) {
      double sum = 0;
      for (int i = 0; i < n; i++) {
        sum += input[i] * math.cos(math.pi / n * (i + 0.5) * k);
      }
      output[k] = sum;
    }
    return output;
  }

  int _nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }
}
