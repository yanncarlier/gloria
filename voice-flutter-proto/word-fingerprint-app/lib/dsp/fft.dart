// Minimal iterative radix-2 Cooley-Tukey FFT for pure-Dart DSP.
// Input length must be a power of two (callers zero-pad frames to fit).

import 'dart:math' as math;

class Complex {
  final double re;
  final double im;

  const Complex(this.re, this.im);

  Complex operator +(Complex o) => Complex(re + o.re, im + o.im);
  Complex operator -(Complex o) => Complex(re - o.re, im - o.im);
  Complex operator *(Complex o) =>
      Complex(re * o.re - im * o.im, re * o.im + im * o.re);

  double get magnitude => math.sqrt(re * re + im * im);
}

/// In-place-style iterative FFT. Returns a new list; does not mutate [input].
/// `input.length` must be a power of two.
List<Complex> fft(List<Complex> input) {
  final n = input.length;
  if (n <= 1) return input;
  assert((n & (n - 1)) == 0, 'FFT length must be a power of 2, got $n');

  final a = List<Complex>.from(input);

  // Bit-reversal permutation.
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      final tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  }

  // Iterative Cooley-Tukey butterfly stages.
  for (int len = 2; len <= n; len <<= 1) {
    final ang = -2 * math.pi / len;
    final wlen = Complex(math.cos(ang), math.sin(ang));
    for (int i = 0; i < n; i += len) {
      var w = const Complex(1, 0);
      final half = len ~/ 2;
      for (int k = 0; k < half; k++) {
        final u = a[i + k];
        final v = a[i + k + half] * w;
        a[i + k] = u + v;
        a[i + k + half] = u - v;
        w = w * wlen;
      }
    }
  }

  return a;
}
