// 识别结果解析（不需要原生库）：用量清单、图例核对、警告和错误的中文说明。

import 'package:flutter_test/flutter_test.dart';
import 'package:pindou_app/result.dart';

void main() {
  test('解析正常结果：用量清单按颗数排序，和图例对照', () {
    final r = Recognition.parse({
      'grid': {'cols': 3, 'rows': 2},
      'codebook': [
        {
          'palette_id': 'MARD-221',
          'code': 'H7',
          'legend_rgb': [0, 0, 0],
        },
        {
          'palette_id': 'MARD-221',
          'code': 'G3',
          'legend_rgb': [236, 197, 172],
        },
      ],
      'cells': [1, 2, 2, 0, 2, 1],
      'uncertain': [4],
      'legend': [
        {'code': 'H7', 'count': 2},
        {'code': 'G3', 'count': 4},
      ],
      'legend_check': {
        'ok': false,
        'diff': [
          {'code': 'G3', 'expected': 4, 'actual': 3},
        ],
      },
      'palette_guess': {'palette_id': 'MARD-221', 'confidence': 1.0},
      'warnings': ['text_evidence_changed:3'],
      'timing_ms': {'grid': 10, 'total': 42},
    });
    expect(r.ok, isTrue);
    expect(r.cols, 3);
    expect(r.beads, 5);
    expect(r.bom.map((b) => b.entry.code), ['G3', 'H7']);
    expect(r.bom.first.count, 3);
    expect(r.bom.first.mismatch, isTrue);
    expect(r.bom.last.mismatch, isFalse);
    expect(r.legendRead, isTrue);
    expect(r.legendOk, isFalse);
    expect(r.uncertain, {4});
    expect(r.brand, 'MARD-221');
    expect(r.engineMs, 42);
    expect(r.warnings.single, contains('3 格'));
  });

  test('解析错误：给出中文说明', () {
    final r = Recognition.parse({
      'error': {'code': 'needs_user_crop', 'message': '图里有多个独立的网格'},
    });
    expect(r.ok, isFalse);
    expect(r.errorText, contains('框选'));
  });

  test('没有读到图例时不做核对', () {
    final r = Recognition.parse({
      'grid': {'cols': 1, 'rows': 1},
      'codebook': [
        {'code': 'A1'},
      ],
      'cells': [1],
      'legend': [],
      'legend_check': {'ok': false, 'diff': [], 'unavailable': true},
    });
    expect(r.legendRead, isFalse);
    expect(r.bom.single.legendCount, isNull);
  });
}
