// 识别引擎：通过 dart:ffi 调用 libbeadcore.so 的 C ABI（core/include/beadcore/beadcore.h）。
// 识别在后台 Isolate 里跑，图片按原尺寸解码成 RGBA，不压缩、不缩小（AGENTS 规则 4）。

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:ui' as ui;

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

final class _BcImage extends Struct {
  external Pointer<Uint8> rgba;
  @Int32()
  external int width;
  @Int32()
  external int height;
  @Int32()
  external int stride;
}

typedef _CreateC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Pointer<Utf8>>);
typedef _DestroyC = Void Function(Pointer<Void>);
typedef _DestroyD = void Function(Pointer<Void>);
typedef _RecognizeC =
    Pointer<Utf8> Function(Pointer<Void>, Pointer<_BcImage>, Int32, Pointer<Utf8>, Pointer<Void>, Pointer<Void>);
typedef _RecognizeD =
    Pointer<Utf8> Function(Pointer<Void>, Pointer<_BcImage>, int, Pointer<Utf8>, Pointer<Void>, Pointer<Void>);
typedef _FreeC = Void Function(Pointer<Utf8>);
typedef _FreeD = void Function(Pointer<Utf8>);
typedef _VersionC = Pointer<Utf8> Function();

/// 解码后的一页图纸（RGBA8888，方向已校正，stride = width * 4）。
class DecodedImage {
  DecodedImage(this.rgba, this.width, this.height);
  final Uint8List rgba;
  final int width, height;
}

/// 一次识别的输入（会整体传进后台 Isolate）。
class _Job {
  _Job(this.modelsDir, this.palettesJson, this.images, this.hint);
  final String modelsDir;
  final String palettesJson;
  final List<DecodedImage> images;
  final String? hint;
}

class RecognizeOutput {
  RecognizeOutput(this.json, this.engineMs);
  final Map<String, dynamic> json;
  final int engineMs; // 包括创建上下文（加载 OCR 模型）在内的原生调用耗时
}

class Engine {
  Engine._(this.modelsDir, this.palettesJson, this.paletteSource, this.version);

  final String modelsDir;
  final String palettesJson;
  final String paletteSource; // 给用户看：用的是哪份色卡
  final String version;

  static const _modelFiles = ['models.json', 'ocr_v3.onnx', 'ocr_v3.json'];

  /// 首次启动把模型从 assets 解压到 App 私有目录（ORT 要从文件加载），并选色卡。
  static Future<Engine> load() async {
    final support = await getApplicationSupportDirectory();
    final dir = Directory('${support.path}/models');
    await dir.create(recursive: true);
    for (final name in _modelFiles) {
      final data = await rootBundle.load('assets/models/$name');
      final f = File('${dir.path}/$name');
      if (!await f.exists() || await f.length() != data.lengthInBytes) {
        await f.writeAsBytes(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes), flush: true);
      }
    }
    // 测试色卡：PO 用 adb 推到 App 的专属外部目录（不进安装包，07「安卓原型」）
    String palettes;
    String source;
    final ext = await getExternalStorageDirectory();
    final test = ext == null ? null : File('${ext.path}/palettes.json');
    if (test != null && await test.exists()) {
      palettes = await test.readAsString();
      source = '测试色卡：${test.path}';
    } else {
      palettes = await rootBundle.loadString('assets/palettes.json');
      source = '内置色卡：12 套已核实来源的国际品牌（不含 MARD 等国内品牌）';
    }
    // 第一次加载识别库会连带加载 34 MB 的 ONNX Runtime，放到后台 Isolate 里，免得卡住界面
    final version = await Isolate.run(
      () => DynamicLibrary.open('libbeadcore.so').lookupFunction<_VersionC, _VersionC>('bc_version')().toDartString(),
    );
    return Engine._(dir.path, palettes, source, version);
  }

  /// 解码：原尺寸（不传 targetWidth / targetHeight），非预乘的 RGBA。
  static Future<DecodedImage> decode(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final data = await image.toByteData(format: ui.ImageByteFormat.rawStraightRgba);
    final out = DecodedImage(data!.buffer.asUint8List(), image.width, image.height);
    image.dispose();
    codec.dispose();
    return out;
  }

  Future<RecognizeOutput> recognize(List<DecodedImage> images, {String? hint}) {
    final job = _Job(modelsDir, palettesJson, images, hint);
    return Isolate.run(() => _run(job));
  }

  static RecognizeOutput _run(_Job job) {
    final lib = DynamicLibrary.open('libbeadcore.so');
    final create = lib.lookupFunction<_CreateC, _CreateC>('bc_create');
    final destroy = lib.lookupFunction<_DestroyC, _DestroyD>('bc_destroy');
    final recognize = lib.lookupFunction<_RecognizeC, _RecognizeD>('bc_recognize');
    final free = lib.lookupFunction<_FreeC, _FreeD>('bc_free');

    final sw = Stopwatch()..start();
    final models = job.modelsDir.toNativeUtf8();
    final palettes = job.palettesJson.toNativeUtf8();
    final err = calloc<Pointer<Utf8>>();
    final imgs = calloc<_BcImage>(job.images.length);
    final buffers = <Pointer<Uint8>>[];
    final hint = job.hint == null ? nullptr : job.hint!.toNativeUtf8();
    try {
      final ctx = create(models, palettes, err);
      if (ctx == nullptr) {
        final msg = err.value == nullptr
            ? '{"error":{"code":"internal","message":"bc_create 失败"}}'
            : err.value.toDartString();
        if (err.value != nullptr) free(err.value);
        return RecognizeOutput(jsonDecode(msg) as Map<String, dynamic>, sw.elapsedMilliseconds);
      }
      for (var i = 0; i < job.images.length; i++) {
        final im = job.images[i];
        final buf = malloc<Uint8>(im.rgba.length);
        buf.asTypedList(im.rgba.length).setAll(0, im.rgba);
        buffers.add(buf);
        imgs[i]
          ..rgba = buf
          ..width = im.width
          ..height = im.height
          ..stride = im.width * 4;
      }
      final res = recognize(ctx, imgs, job.images.length, hint, nullptr, nullptr);
      final text = res.toDartString();
      free(res);
      destroy(ctx);
      return RecognizeOutput(jsonDecode(text) as Map<String, dynamic>, sw.elapsedMilliseconds);
    } finally {
      for (final b in buffers) {
        malloc.free(b);
      }
      calloc.free(imgs);
      calloc.free(err);
      malloc.free(models);
      malloc.free(palettes);
      if (hint != nullptr) malloc.free(hint);
    }
  }
}
