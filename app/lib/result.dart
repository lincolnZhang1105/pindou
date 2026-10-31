// RecognitionResult（03 §3.2）里原型用得到的字段。

import 'dart:ui';

class BookEntry {
  BookEntry(this.code, this.color);
  final String code;
  final Color color;
}

class BomItem {
  BomItem(this.index, this.entry, this.count, this.legendCount);
  final int index; // codebook 下标（从 1 开始，和 cells 一致）
  final BookEntry entry;
  final int count;
  final int? legendCount; // 图例上写的颗数（读到时）
  bool get mismatch => legendCount != null && legendCount != count;
}

const _warnings = {
  'cell_too_small_for_text': '格子太小，格内文字读不出来，只按颜色识别',
  'duplicate_legend_code': '图例里有重复的色号',
  'legend_count_adjusted': '按图例颗数调整了部分格子',
  'pages_order_unverified': '分页图的拼接顺序是按版式猜的，请核对',
  'palette_hint_not_found': '没有找到指定的色卡',
  'text_evidence_changed': '按格内色号文字改判了部分格子',
};

const _errors = {
  'no_grid_found': '没有找到网格',
  'needs_user_crop': '图里像是有多个图案，需要先框选其中一个（原型还不支持框选）',
  'invalid_input': '输入有问题',
  'oom': '内存不足',
  'internal': '识别程序内部出错',
};

class Recognition {
  Recognition._(this.raw);

  final Map<String, dynamic> raw;
  String? errorCode, errorMessage;
  int cols = 0, rows = 0;
  List<BookEntry> book = [];
  List<int> cells = [];
  Set<int> uncertain = {};
  List<String> warnings = [];
  String? brand;
  double brandConfidence = 0; // palette_guess.confidence：低于 0.5 时界面上显示「未确定」
  bool legendRead = false, legendOk = false;
  int engineMs = 0; // 引擎自己报的识别耗时（timing_ms.total）
  List<BomItem> bom = [];

  bool get ok => errorCode == null;
  int get beads => bom.fold(0, (s, b) => s + b.count);
  String get errorText => _errors[errorCode] ?? errorCode ?? '';

  static Recognition parse(Map<String, dynamic> j) {
    final r = Recognition._(j);
    final err = j['error'];
    if (err is Map) {
      r.errorCode = err['code'] as String?;
      r.errorMessage = err['message'] as String?;
      return r;
    }
    r.cols = j['grid']['cols'] as int;
    r.rows = j['grid']['rows'] as int;
    r.book = [for (final e in (j['codebook'] as List)) BookEntry(e['code'] as String, _color(e['legend_rgb']))];
    r.cells = (j['cells'] as List).cast<int>();
    r.uncertain = ((j['uncertain'] as List?) ?? const []).cast<int>().toSet();
    r.warnings = [for (final w in ((j['warnings'] as List?) ?? const []).cast<String>()) _warningText(w)];
    r.brand = (j['palette_guess'] as Map?)?['palette_id'] as String?;
    r.brandConfidence = ((j['palette_guess'] as Map?)?['confidence'] as num?)?.toDouble() ?? 0;
    final legend = <String, int>{
      for (final it in ((j['legend'] as List?) ?? const [])) it['code'] as String: (it['count'] as num).toInt(),
    };
    final check = j['legend_check'] as Map?;
    r.legendRead = legend.isNotEmpty && check?['unavailable'] != true;
    r.legendOk = r.legendRead && check?['ok'] == true;
    r.engineMs = ((j['timing_ms'] as Map?)?['total'] as num?)?.toInt() ?? 0;
    final counts = List<int>.filled(r.book.length + 1, 0);
    for (final c in r.cells) {
      if (c > 0) counts[c]++;
    }
    r.bom = [
      for (var k = 1; k <= r.book.length; k++)
        if (counts[k] > 0) BomItem(k, r.book[k - 1], counts[k], legend[r.book[k - 1].code]),
    ]..sort((a, b) => b.count.compareTo(a.count));
    return r;
  }

  static Color _color(dynamic rgb) {
    if (rgb is List && rgb.length >= 3) {
      return Color.fromARGB(255, (rgb[0] as num).toInt(), (rgb[1] as num).toInt(), (rgb[2] as num).toInt());
    }
    return const Color(0xFFC8C8C8);
  }

  static String _warningText(String w) {
    final i = w.indexOf(':');
    final key = i < 0 ? w : w.substring(0, i);
    final text = _warnings[key] ?? w;
    return i < 0 ? text : '$text（${w.substring(i + 1)} 格）';
  }
}
