// 重画的图纸：可缩放、平移；只画屏幕里看得见的格子；放大到格子够大时写出色号。
// 选中一个色号时，其他格子变淡（高亮跟拼的最小版本）；不确定的格子描红框。

import 'package:flutter/material.dart';

import 'result.dart';

const double kCell = 20; // 一格的逻辑像素（缩放前）

class PatternView extends StatefulWidget {
  const PatternView({super.key, required this.result, this.highlight});

  final Recognition result;
  final int? highlight; // codebook 下标；null 表示不高亮

  @override
  State<PatternView> createState() => _PatternViewState();
}

class _PatternViewState extends State<PatternView> {
  final _controller = TransformationController();
  bool _fitted = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final r = widget.result;
    final size = Size(r.cols * kCell, r.rows * kCell);
    return LayoutBuilder(
      builder: (context, box) {
        if (!_fitted && box.maxWidth.isFinite && box.maxHeight.isFinite) {
          // 初始缩放到整张图纸放得下
          final s = (box.maxWidth / size.width).clamp(0.02, 1.0).toDouble();
          final s2 = (box.maxHeight / size.height).clamp(0.02, 1.0).toDouble();
          _controller.value = Matrix4.identity()..scaleByDouble(s < s2 ? s : s2, s < s2 ? s : s2, 1, 1);
          _fitted = true;
        }
        return InteractiveViewer(
          transformationController: _controller,
          constrained: false,
          minScale: 0.02,
          maxScale: 6,
          boundaryMargin: const EdgeInsets.all(200),
          child: ValueListenableBuilder<Matrix4>(
            valueListenable: _controller,
            builder: (context, m, _) {
              final scale = m.getMaxScaleOnAxis();
              final inv = Matrix4.inverted(m);
              final tl = MatrixUtils.transformPoint(inv, Offset.zero);
              final br = MatrixUtils.transformPoint(inv, Offset(box.maxWidth, box.maxHeight));
              return CustomPaint(
                size: size,
                painter: _PatternPainter(r, widget.highlight, scale, Rect.fromPoints(tl, br)),
              );
            },
          ),
        );
      },
    );
  }
}

class _PatternPainter extends CustomPainter {
  _PatternPainter(this.r, this.highlight, this.scale, this.visible);

  final Recognition r;
  final int? highlight;
  final double scale;
  final Rect visible;

  @override
  void paint(Canvas canvas, Size size) {
    final c0 = (visible.left / kCell).floor().clamp(0, r.cols);
    final c1 = (visible.right / kCell).ceil().clamp(0, r.cols);
    final r0 = (visible.top / kCell).floor().clamp(0, r.rows);
    final r1 = (visible.bottom / kCell).ceil().clamp(0, r.rows);
    final fill = Paint();
    final showText = kCell * scale >= 26 && (c1 - c0) * (r1 - r0) <= 2500;
    for (var y = r0; y < r1; y++) {
      for (var x = c0; x < c1; x++) {
        final k = r.cells[y * r.cols + x];
        final rect = Rect.fromLTWH(x * kCell, y * kCell, kCell, kCell);
        var color = k == 0 ? Colors.white : r.book[k - 1].color;
        if (highlight != null && k != highlight) color = Color.lerp(color, Colors.white, 0.85)!;
        fill.color = color;
        canvas.drawRect(rect, fill);
        if (showText && k != 0 && (highlight == null || k == highlight)) {
          final lum = color.computeLuminance();
          final tp = TextPainter(
            text: TextSpan(
              text: r.book[k - 1].code,
              style: TextStyle(fontSize: 6.5, color: lum < 0.4 ? Colors.white : Colors.black87),
            ),
            textDirection: TextDirection.ltr,
          )..layout(maxWidth: kCell);
          tp.paint(canvas, rect.center - Offset(tp.width / 2, tp.height / 2));
        }
      }
    }
    // 网格线：每格细线，每 5 格粗一点
    final thin = Paint()
      ..color = const Color(0x33000000)
      ..strokeWidth = 0.5 / scale.clamp(0.5, 6);
    final bold = Paint()
      ..color = const Color(0x88000000)
      ..strokeWidth = 1.2 / scale.clamp(0.5, 6);
    for (var x = c0; x <= c1; x++) {
      canvas.drawLine(Offset(x * kCell, r0 * kCell), Offset(x * kCell, r1 * kCell), x % 5 == 0 ? bold : thin);
    }
    for (var y = r0; y <= r1; y++) {
      canvas.drawLine(Offset(c0 * kCell, y * kCell), Offset(c1 * kCell, y * kCell), y % 5 == 0 ? bold : thin);
    }
    // 不确定的格子
    if (r.uncertain.isNotEmpty) {
      final red = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2 / scale.clamp(0.5, 6);
      for (final i in r.uncertain) {
        final x = i % r.cols, y = i ~/ r.cols;
        if (x < c0 || x >= c1 || y < r0 || y >= r1) continue;
        canvas.drawRect(Rect.fromLTWH(x * kCell, y * kCell, kCell, kCell).deflate(1), red);
      }
    }
  }

  @override
  bool shouldRepaint(_PatternPainter old) =>
      old.r != r || old.highlight != highlight || old.scale != scale || old.visible != visible;
}
