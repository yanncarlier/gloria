import 'package:flutter_test/flutter_test.dart';

import 'package:word_fingerprint_app/dsp/fingerprint.dart';
import 'package:word_fingerprint_app/dsp/mfcc.dart';

void main() {
  test('identical voice sequences are accepted as the same word', () {
    final sequence = [
      [0.9, 0.2, -0.3, 0.1],
      [0.8, 0.1, -0.2, 0.0],
      [0.7, 0.2, -0.1, 0.1],
    ];

    final comparison = compareVoiceSequences(sequence, sequence);

    expect(comparison.isSameWord, isTrue);
    expect(comparison.verdict, 'SAME WORD');
  });

  test('slightly perturbed same-word sequences are accepted', () {
    final first = [
      [0.9, 0.2, -0.3, 0.1],
      [0.8, 0.1, -0.2, 0.0],
      [0.7, 0.2, -0.1, 0.1],
    ];
    final second = [
      [0.88, 0.18, -0.32, 0.12],
      [0.82, 0.12, -0.19, -0.01],
      [0.72, 0.21, -0.09, 0.11],
    ];

    final comparison = compareVoiceSequences(first, second);

    expect(comparison.isSameWord, isTrue);
    expect(comparison.verdict, 'SAME WORD');
  });

  test('mfcc normalization removes coefficient drift across frames', () {
    final extractor = MfccExtractor();
    final sequence = [
      [1.0, 2.0],
      [1.4, 2.6],
      [0.6, 1.4],
    ];

    final normalized = extractor.normalizeMfccSequence(sequence);

    for (int coeff = 0; coeff < normalized[0].length; coeff++) {
      double sum = 0;
      for (final frame in normalized) {
        sum += frame[coeff];
      }
      final mean = sum / normalized.length;
      expect(mean.abs(), lessThan(1e-9));
    }
  });

  test('distinct voice sequences are marked as different', () {
    final first = [
      [0.9, 0.2, -0.3, 0.1],
      [0.8, 0.1, -0.2, 0.0],
      [0.7, 0.2, -0.1, 0.1],
    ];
    final second = [
      [-0.8, 0.7, 0.3, -0.6],
      [-0.9, 0.8, 0.2, -0.7],
      [-0.7, 0.6, 0.4, -0.5],
    ];

    final comparison = compareVoiceSequences(first, second);

    expect(comparison.isSameWord, isFalse);
    expect(comparison.verdict, 'DIFFERENT');
  });
}
