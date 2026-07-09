// Very simple Flutter app: record a few takes of the same spoken word,
// then fingerprint and compare them (DTW distance + SimHash fingerprint +
// SHA-256 proof) to decide whether they're the same word.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

import 'dsp/wav_reader.dart';
import 'dsp/mfcc.dart';
import 'dsp/fingerprint.dart';

void main() {
  runApp(const WordFingerprintApp());
}

class WordFingerprintApp extends StatelessWidget {
  const WordFingerprintApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Word Fingerprint',
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const HomePage(),
    );
  }
}

class WordSample {
  final String name;
  final String filePath;
  List<List<double>>? mfccSeq;
  Fingerprint? fingerprint;

  WordSample({required this.name, required this.filePath});
}

class ComparisonRow {
  final String nameA;
  final String nameB;
  final double dtwDistance;
  final int hammingDistance;
  final String verdict;

  ComparisonRow({
    required this.nameA,
    required this.nameB,
    required this.dtwDistance,
    required this.hammingDistance,
    required this.verdict,
  });
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final AudioRecorder _recorder = AudioRecorder();
  final MfccExtractor _mfccExtractor = MfccExtractor();
  final List<WordSample> _samples = [];

  bool _isRecording = false;
  bool _isProcessing = false;
  List<ComparisonRow> _results = [];

  static const int _sampleRate = 16000;
  // Tune these two thresholds against your own recordings.
  static const double _dtwSameWordThreshold = 45.0;
  static const int _hammingSameWordThreshold = 20;

  @override
  void dispose() {
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final path = await _recorder.stop();
      setState(() => _isRecording = false);
      if (path != null) {
        final name = 'take_${_samples.length + 1}';
        setState(() => _samples.add(WordSample(name: name, filePath: path)));
      }
      return;
    }

    var hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      final status = await Permission.microphone.request();
      hasPermission = status.isGranted;
    }
    if (!hasPermission) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Microphone permission is required.')),
        );
      }
      return;
    }

    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/word_${DateTime.now().millisecondsSinceEpoch}.wav';

    await _recorder.start(
      const RecordConfig(
        encoder: AudioEncoder.wav,
        sampleRate: _sampleRate,
        numChannels: 1,
      ),
      path: path,
    );
    setState(() => _isRecording = true);
  }

  Future<void> _removeSample(int index) async {
    final sample = _samples[index];
    try {
      final f = File(sample.filePath);
      if (await f.exists()) await f.delete();
    } catch (_) {
      // Non-fatal: file cleanup failures shouldn't block the UI.
    }
    setState(() {
      _samples.removeAt(index);
      _results = [];
    });
  }

  Future<void> _compareAll() async {
    if (_samples.length < 2) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Record at least 2 takes first.')),
      );
      return;
    }

    setState(() => _isProcessing = true);

    try {
      for (final sample in _samples) {
        if (sample.mfccSeq != null) continue;
        final bytes = await File(sample.filePath).readAsBytes();
        final wav = readWavFile(bytes);
        final mfccSeq = _mfccExtractor.extract(wav.samples, wav.sampleRate);
        final pooled = poolFeatures(mfccSeq);
        final fp = computeFingerprint(pooled);
        sample.mfccSeq = mfccSeq;
        sample.fingerprint = fp;
      }

      final rows = <ComparisonRow>[];
      for (int i = 0; i < _samples.length; i++) {
        for (int j = i + 1; j < _samples.length; j++) {
          final a = _samples[i];
          final b = _samples[j];
          final dtwD = dtwDistance(a.mfccSeq!, b.mfccSeq!);
          final hamD = hammingDistance(a.fingerprint!.bits, b.fingerprint!.bits);

          final sameByDtw = dtwD < _dtwSameWordThreshold;
          final sameByHash = hamD < _hammingSameWordThreshold;
          String verdict;
          if (sameByDtw && sameByHash) {
            verdict = 'SAME WORD';
          } else if (sameByDtw || sameByHash) {
            verdict = 'UNCERTAIN';
          } else {
            verdict = 'DIFFERENT';
          }

          rows.add(ComparisonRow(
            nameA: a.name,
            nameB: b.name,
            dtwDistance: dtwD,
            hammingDistance: hamD,
            verdict: verdict,
          ));
        }
      }

      setState(() => _results = rows);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error processing audio: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Color _verdictColor(String verdict) {
    switch (verdict) {
      case 'SAME WORD':
        return Colors.green;
      case 'DIFFERENT':
        return Colors.red;
      default:
        return Colors.orange;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Word Fingerprint')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton.icon(
              onPressed: _toggleRecording,
              icon: Icon(_isRecording ? Icons.stop : Icons.mic),
              label: Text(_isRecording ? 'Stop Recording' : 'Record New Take'),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isRecording ? Colors.red : null,
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            const SizedBox(height: 16),
            Text('Recorded takes (${_samples.length})',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Expanded(
              flex: 2,
              child: _samples.isEmpty
                  ? const Center(child: Text('No takes recorded yet.'))
                  : ListView.builder(
                      itemCount: _samples.length,
                      itemBuilder: (context, index) {
                        final s = _samples[index];
                        return Card(
                          child: ListTile(
                            title: Text(s.name),
                            subtitle: s.fingerprint != null
                                ? Text(
                                    'fp: ${s.fingerprint!.hex.substring(0, 16)}...',
                                    style: const TextStyle(
                                        fontFamily: 'monospace', fontSize: 12),
                                  )
                                : const Text('Not processed yet'),
                            trailing: IconButton(
                              icon: const Icon(Icons.delete_outline),
                              onPressed: () => _removeSample(index),
                            ),
                          ),
                        );
                      },
                    ),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: _isProcessing ? null : _compareAll,
              icon: _isProcessing
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.compare_arrows),
              label: Text(_isProcessing ? 'Processing...' : 'Compare All Takes'),
            ),
            const SizedBox(height: 16),
            if (_results.isNotEmpty) ...[
              Text('Results', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Expanded(
                flex: 3,
                child: ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, index) {
                    final r = _results[index];
                    return Card(
                      child: ListTile(
                        title: Text('${r.nameA} vs ${r.nameB}'),
                        subtitle: Text(
                          'DTW: ${r.dtwDistance.toStringAsFixed(2)}   '
                          'Hamming: ${r.hammingDistance}/128',
                        ),
                        trailing: Chip(
                          label: Text(
                            r.verdict,
                            style: const TextStyle(color: Colors.white, fontSize: 12),
                          ),
                          backgroundColor: _verdictColor(r.verdict),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
