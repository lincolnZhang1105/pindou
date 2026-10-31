plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// 打包哪些 ABI：默认只有 arm64（给手机）；tools/build-app.sh 用 -Ppindou.abis=arm64-v8a,x86_64 加上模拟器用的 x86_64
val pindouAbis = ((project.findProperty("pindou.abis") as String?) ?: "arm64-v8a").split(",").map { it.trim() }

android {
    namespace = "dev.pindou.pindou_app"
    compileSdk = flutter.compileSdkVersion
    // 识别库 libbeadcore.so 是在 WSL 里用 NDK r27c 预先编好的（tools/build-android.sh），不在这里编译。
    // 但 Flutter 的 Gradle 插件在配置阶段仍要求 SDK 里装有它默认的 NDK 版本（3.47.9 是 28.2.13676358）
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // 原型专用的应用 ID；正式包名在阶段 2 定稿（07 T2.5）
        applicationId = "dev.pindou.proto"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 26  // 和 libbeadcore.so 编译时的 ANDROID_PLATFORM 一致
        ndk {
            abiFilters += pindouAbis  // 只打包编了识别库的 ABI
        }
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // path_provider 间接引入的 jni 包会带上所有 ABI 的 libdartjni.so；没编识别库的 ABI 排除掉，
    // 否则 32 位手机能装上但一识别就崩，排除后在这种手机上直接装不上
    packaging {
        jniLibs {
            excludes += listOf("armeabi-v7a", "x86_64", "x86", "arm64-v8a").filter { it !in pindouAbis }.map { "lib/$it/**" }
        }
    }

    buildTypes {
        release {
            // 原型只用 adb 装在 PO 手机上，用 debug 密钥签名，不能上架
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
