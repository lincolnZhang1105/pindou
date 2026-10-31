// 豆伴安卓原型（07「例外：安卓原型」）：导入图纸 → 识别 → 看结果。
// 不联网、不集成任何 SDK、不申请相册 / 存储 / 蓝牙权限；只装在 PO 自己的手机上。

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'engine.dart';
import 'result.dart';
import 'result_page.dart';

void main() => runApp(const PindouApp());

class PindouApp extends StatelessWidget {
  const PindouApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: '豆伴（原型）',
    theme: ThemeData(colorSchemeSeed: const Color(0xFFE07A5F), useMaterial3: true),
    home: const HomePage(),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Engine? _engine;
  String? _loadError;
  String? _busy; // 正在做什么（显示在按钮下面）

  @override
  void initState() {
    super.initState();
    Engine.load().then((e) => setState(() => _engine = e), onError: (Object e) => setState(() => _loadError = '$e'));
  }

  Future<void> _pick() async {
    // 系统选择器（Android 13 起是 Photo Picker），不申请权限；不传 maxWidth / imageQuality，拿到的是原图
    final files = await ImagePicker().pickMultiImage(requestFullMetadata: false);
    if (files.isEmpty) return;
    final bytes = [for (final f in files) await f.readAsBytes()];
    await _run(bytes, files.length == 1 ? files.first.name : '${files.length} 张（按选择顺序拼接）');
  }

  Future<void> _benchmark() async {
    final data = await rootBundle.load('assets/bench_104.png');
    await _run([data.buffer.asUint8List()], '内置测速图 104×104');
  }

  Future<void> _run(List<Uint8List> bytes, String title) async {
    final engine = _engine;
    if (engine == null) return;
    final sw = Stopwatch()..start();
    try {
      setState(() => _busy = '解码图片…');
      final images = [for (final b in bytes) await Engine.decode(b)];
      setState(() => _busy = '识别中…（${images.map((i) => '${i.width}×${i.height}').join('，')}）');
      final out = await engine.recognize(images);
      final r = Recognition.parse(out.json);
      if (!mounted) return;
      setState(() => _busy = null);
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultPage(
            result: r,
            title: title,
            totalMs: sw.elapsedMilliseconds,
            sources: [
              for (var i = 0; i < images.length; i++)
                '${images[i].width}×${images[i].height}，${(bytes[i].length / 1024).toStringAsFixed(0)} KB',
            ].join('；'),
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() => _busy = null);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('出错了：$e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final e = _engine;
    return Scaffold(
      appBar: AppBar(title: const Text('豆伴（原型）')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('在小红书里长按图纸「保存图片」，再从这里导入。分页图一次多选，按页码顺序选。', style: TextStyle(color: Colors.black54)),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: e == null || _busy != null ? null : _pick,
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('导入图纸'),
            ),
            const SizedBox(height: 8),
            OutlinedButton.icon(
              onPressed: e == null || _busy != null ? null : _benchmark,
              icon: const Icon(Icons.speed),
              label: const Text('用内置测速图测一下速度'),
            ),
            const SizedBox(height: 16),
            if (_busy != null)
              Row(
                children: [
                  const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                  const SizedBox(width: 10),
                  Expanded(child: Text(_busy!)),
                ],
              ),
            if (_loadError != null) Text('引擎加载失败：$_loadError', style: const TextStyle(color: Colors.red)),
            const Spacer(),
            if (e != null)
              Text('引擎 ${e.version} · ${e.paletteSource}', style: const TextStyle(fontSize: 11, color: Colors.black45)),
            const Text('原型：不联网，识别只在手机上做，图片不会上传。', style: TextStyle(fontSize: 11, color: Colors.black45)),
          ],
        ),
      ),
    );
  }
}
