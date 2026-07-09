// Minimal RIFF/WAVE parser for 16-bit PCM files, robust to extra chunks
// (e.g. LIST/INFO metadata some recorders add before the 'data' chunk).

import 'dart:typed_data';

class WavData {
  final int sampleRate;
  final int numChannels;
  final List<double> samples; // mono, normalized to [-1.0, 1.0]

  WavData(this.sampleRate, this.numChannels, this.samples);
}

WavData readWavFile(Uint8List bytes) {
  if (bytes.length < 44) {
    throw FormatException('File too short to be a valid WAV file.');
  }
  final byteData = ByteData.sublistView(bytes);

  final riff = String.fromCharCodes(bytes.sublist(0, 4));
  final wave = String.fromCharCodes(bytes.sublist(8, 12));
  if (riff != 'RIFF' || wave != 'WAVE') {
    throw FormatException('Not a valid RIFF/WAVE file.');
  }

  int pos = 12;
  int sampleRate = 16000;
  int numChannels = 1;
  int bitsPerSample = 16;
  int dataOffset = -1;
  int dataSize = 0;

  while (pos + 8 <= bytes.length) {
    final chunkId = String.fromCharCodes(bytes.sublist(pos, pos + 4));
    final chunkSize = byteData.getUint32(pos + 4, Endian.little);
    final chunkDataStart = pos + 8;

    if (chunkId == 'fmt ') {
      numChannels = byteData.getUint16(chunkDataStart + 2, Endian.little);
      sampleRate = byteData.getUint32(chunkDataStart + 4, Endian.little);
      bitsPerSample = byteData.getUint16(chunkDataStart + 14, Endian.little);
    } else if (chunkId == 'data') {
      dataOffset = chunkDataStart;
      dataSize = chunkSize;
    }

    // Chunks are word-aligned; skip a padding byte if chunkSize is odd.
    pos = chunkDataStart + chunkSize + (chunkSize % 2);
  }

  if (dataOffset == -1) {
    throw FormatException('No data chunk found in WAV file.');
  }
  if (bitsPerSample != 16) {
    throw FormatException(
        'Only 16-bit PCM WAV is supported (got $bitsPerSample-bit).');
  }

  final bytesPerSample = bitsPerSample ~/ 8;
  final numSamples = dataSize ~/ bytesPerSample;
  final maxAvailable = ((bytes.length - dataOffset) ~/ bytesPerSample);
  final safeNumSamples = numSamples > maxAvailable ? maxAvailable : numSamples;

  final allSamples = List<double>.filled(safeNumSamples, 0);
  for (int i = 0; i < safeNumSamples; i++) {
    final sampleInt =
        byteData.getInt16(dataOffset + i * bytesPerSample, Endian.little);
    allSamples[i] = sampleInt / 32768.0;
  }

  List<double> mono;
  if (numChannels > 1) {
    final frames = safeNumSamples ~/ numChannels;
    mono = List<double>.filled(frames, 0);
    for (int f = 0; f < frames; f++) {
      double sum = 0;
      for (int c = 0; c < numChannels; c++) {
        sum += allSamples[f * numChannels + c];
      }
      mono[f] = sum / numChannels;
    }
  } else {
    mono = allSamples;
  }

  return WavData(sampleRate, numChannels, mono);
}
