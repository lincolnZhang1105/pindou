// 结果页：概览、图例核对、重画的图纸（可缩放）、用量清单（点一个色号高亮它的格子）。

import 'package:flutter/material.dart';

import 'pattern_view.dart';
import 'result.dart';

class ResultPage extends StatefulWidget {
  const ResultPage({
    super.key,
    required this.result,
    required this.title,
    required this.totalMs,
    required this.sources,
  });

  final Recognition result;
  final String title;
  final int totalMs; // 从点下按钮到出结果（含解码、加载模型）
  final String sources; // 导入的原图：每张的分辨率和文件大小（核对导入时有没有被压缩）

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  int? _highlight;

  @override
  Widget build(BuildContext context) {
    final r = widget.result;
    return Scaffold(
      appBar: AppBar(title: Text(widget.title, overflow: TextOverflow.ellipsis)),
      body: r.ok ? _body(r) : _error(r),
    );
  }

  Widget _error(Recognition r) => Padding(
    padding: const EdgeInsets.all(24),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          r.errorText,
          style: const TextStyle(fontSize: 18, color: Colors.red, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 8),
        Text(r.errorMessage ?? '', style: const TextStyle(color: Colors.black54)),
      ],
    ),
  );

  Widget _body(Recognition r) {
    final sel = _highlight == null ? null : r.bom.firstWhere((b) => b.index == _highlight);
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
          child: Wrap(
            spacing: 8,
            runSpacing: 6,
            children: [
              _chip('${r.cols} × ${r.rows}', '网格'),
              _chip('${r.beads}', '豆子'),
              _chip('${r.bom.length}', '色号'),
              _chip(r.brand == null || r.brandConfidence < 0.5 ? '未确定' : r.brand!, '品牌'),
              _chip('${r.engineMs} ms', '识别（总 ${widget.totalMs} ms）'),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Row(
            children: [
              Expanded(child: _legendStatus(r)),
              if (r.uncertain.isNotEmpty)
                Text('不确定 ${r.uncertain.length} 格', style: const TextStyle(color: Colors.red)),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Text('原图：${widget.sources}', style: const TextStyle(color: Colors.black45, fontSize: 12)),
        ),
        for (final w in r.warnings)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Text('· $w', style: const TextStyle(color: Color(0xFFB26A00), fontSize: 12)),
          ),
        Expanded(
          flex: 3,
          child: Container(
            margin: const EdgeInsets.all(8),
            decoration: BoxDecoration(border: Border.all(color: Colors.black12)),
            clipBehavior: Clip.hardEdge,
            child: PatternView(result: r, highlight: _highlight),
          ),
        ),
        if (sel != null)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Text('只显示 ${sel.entry.code}：${sel.count} 颗（再点一次取消）'),
          ),
        Expanded(
          flex: 2,
          child: ListView(
            children: [
              for (final b in r.bom)
                ListTile(
                  dense: true,
                  selected: b.index == _highlight,
                  leading: Container(
                    width: 22,
                    height: 22,
                    decoration: BoxDecoration(
                      color: b.entry.color,
                      border: Border.all(color: Colors.black26),
                    ),
                  ),
                  title: Text(b.entry.code),
                  trailing: Text(
                    b.legendCount == null ? '${b.count}' : '${b.count}（图例 ${b.legendCount}）',
                    style: TextStyle(
                      color: b.mismatch ? Colors.red : null,
                      fontWeight: b.mismatch ? FontWeight.w600 : null,
                    ),
                  ),
                  onTap: () => setState(() => _highlight = _highlight == b.index ? null : b.index),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _legendStatus(Recognition r) {
    if (!r.legendRead) return const Text('没有读到图例，无法核对', style: TextStyle(color: Color(0xFFB26A00)));
    if (r.legendOk) {
      return const Text(
        '与图例完全一致',
        style: TextStyle(color: Color(0xFF1A7F37), fontWeight: FontWeight.w600),
      );
    }
    final n = r.bom.where((b) => b.mismatch).length;
    return Text(
      '有 $n 个色号的颗数和图例对不上',
      style: const TextStyle(color: Colors.red, fontWeight: FontWeight.w600),
    );
  }

  Widget _chip(String value, String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
    decoration: BoxDecoration(
      border: Border.all(color: Colors.black12),
      borderRadius: BorderRadius.circular(8),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(value, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.black54)),
      ],
    ),
  );
}
