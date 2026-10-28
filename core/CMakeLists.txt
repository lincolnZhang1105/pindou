cmake_minimum_required(VERSION 3.21)
project(beadcore VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# beadcore：识别与生成核心（03 §3）。对外只暴露 include/beadcore/beadcore.h 的 C ABI。
# 依赖由 core/scripts/fetch_deps.py 准备到 BC_DEPS_DIR（默认仓库根目录的 .deps/）。
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(BC_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(BC_DEPS_DIR "$ENV{BC_DEPS_DIR}" CACHE PATH "fetch_deps.py 的输出目录")
if(NOT BC_DEPS_DIR)
  set(BC_DEPS_DIR "${BC_REPO_ROOT}/.deps" CACHE PATH "" FORCE)
endif()

option(BC_BUILD_CLI "编译 beadcore-cli（benchmark 和真机测速用）" ON)
option(BC_BUILD_TESTS "编译单元测试" ON)

# 平台标签，和 fetch_deps.py 的 default_platform() 保持一致
if(NOT BC_PLATFORM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _arch)
  if(_arch MATCHES "amd64|x86_64")
    set(_arch x64)
  elseif(_arch MATCHES "aarch64|arm64")
    set(_arch arm64)
  endif()
  if(ANDROID)
    set(BC_PLATFORM "android-${ANDROID_ABI}")
  elseif(WIN32)
    set(BC_PLATFORM "windows-${_arch}-msvc")
  elseif(APPLE)
    set(BC_PLATFORM "macos-${_arch}")
  else()
    set(BC_PLATFORM "linux-${_arch}")
  endif()
endif()
message(STATUS "beadcore: BC_PLATFORM=${BC_PLATFORM} BC_DEPS_DIR=${BC_DEPS_DIR}")

set(OpenCV_STATIC ON)  # fetch_deps.py 只编静态库
# Android 安装布局是 sdk/native/jni/；NDK 工具链默认只在 sysroot 里找包，所以要 NO_CMAKE_FIND_ROOT_PATH
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc
  HINTS "${BC_DEPS_DIR}/${BC_PLATFORM}/opencv" "${BC_DEPS_DIR}/${BC_PLATFORM}/opencv/sdk/native/jni"
  NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "beadcore: OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

# ONNX Runtime（端侧推理，03 §1）。桌面用官方预编译包；App 必须用 --no_telemetry 的定制构建（官方包默认开启遥测）
option(BC_WITH_ORT "链接 ONNX Runtime（关闭时只走颜色路径）" ON)
set(BC_ORT_DIR "${BC_DEPS_DIR}/${BC_PLATFORM}/onnxruntime" CACHE PATH "ONNX Runtime 目录（含 include/ 和 lib/）")
if(BC_WITH_ORT)
  if(NOT EXISTS "${BC_ORT_DIR}/include/onnxruntime_cxx_api.h" AND NOT EXISTS "${BC_ORT_DIR}/headers/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "找不到 ONNX Runtime：${BC_ORT_DIR}（运行 python core/scripts/fetch_deps.py，或 -DBC_WITH_ORT=OFF）")
  endif()
  add_library(bc_onnxruntime SHARED IMPORTED)
  if(ANDROID)
    # 官方 AAR 的布局：headers/ 和 jni/<abi>/libonnxruntime.so
    set_target_properties(bc_onnxruntime PROPERTIES
      IMPORTED_LOCATION "${BC_ORT_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so"
      IMPORTED_NO_SONAME TRUE)
    set(_ort_include "${BC_ORT_DIR}/headers")
  elseif(WIN32)
    set_target_properties(bc_onnxruntime PROPERTIES
      IMPORTED_LOCATION "${BC_ORT_DIR}/lib/onnxruntime.dll"
      IMPORTED_IMPLIB "${BC_ORT_DIR}/lib/onnxruntime.lib")
  elseif(APPLE)
    file(GLOB _ort_lib "${BC_ORT_DIR}/lib/libonnxruntime*.dylib")
    list(GET _ort_lib 0 _ort_lib)
    set_target_properties(bc_onnxruntime PROPERTIES IMPORTED_LOCATION "${_ort_lib}")
  else()
    set_target_properties(bc_onnxruntime PROPERTIES IMPORTED_LOCATION "${BC_ORT_DIR}/lib/libonnxruntime.so")
  endif()
  if(NOT _ort_include)
    set(_ort_include "${BC_ORT_DIR}/include")
  endif()
  set_target_properties(bc_onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_ort_include}")
endif()

set(BC_THIRD_PARTY_INCLUDE "${BC_DEPS_DIR}/src/include")
if(NOT EXISTS "${BC_THIRD_PARTY_INCLUDE}/nlohmann/json.hpp")
  message(FATAL_ERROR "缺少 nlohmann/json.hpp，请先运行 python core/scripts/fetch_deps.py")
endif()

# 编译选项：源码含中文注释，MSVC 必须 /utf-8；不开 fast-math，保证各平台结果确定
if(MSVC)
  add_compile_options(/utf-8 /permissive- /W4 /wd4100 /wd4244 /wd4267 /wd4389)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
endif()

# ---------------------------------------------------------------------------
# 核心库
# ---------------------------------------------------------------------------
file(GLOB_RECURSE BC_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(beadcore STATIC ${BC_SOURCES})
target_include_directories(beadcore
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${BC_THIRD_PARTY_INCLUDE}")
target_link_libraries(beadcore PUBLIC ${OpenCV_LIBS})
if(BC_WITH_ORT)
  target_link_libraries(beadcore PUBLIC bc_onnxruntime)
  target_compile_definitions(beadcore PUBLIC BC_HAVE_ORT=1)
endif()

# Windows 上把 onnxruntime.dll 拷到可执行文件旁边
function(bc_copy_runtime target)
  if(WIN32 AND BC_WITH_ORT)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${BC_ORT_DIR}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:${target}>")
  endif()
endfunction()
target_compile_definitions(beadcore PRIVATE BC_VERSION_STRING="${PROJECT_VERSION}")

# 内部头文件给测试和 CLI 用（不对外）
add_library(beadcore_internal INTERFACE)
target_include_directories(beadcore_internal INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/src" "${BC_THIRD_PARTY_INCLUDE}")
target_link_libraries(beadcore_internal INTERFACE beadcore)

# ---------------------------------------------------------------------------
# CLI：只在桌面和真机测速时用。图片解码只在这里做（App 里由平台层解码，03 §3.1）
# ---------------------------------------------------------------------------
if(BC_BUILD_CLI)
  set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_FUZZTEST OFF CACHE BOOL "" FORCE)
  set(WEBP_LINK_STATIC ON CACHE BOOL "" FORCE)
  add_subdirectory("${BC_DEPS_DIR}/src/libwebp" "${CMAKE_BINARY_DIR}/_deps/libwebp" EXCLUDE_FROM_ALL SYSTEM)

  add_library(bc_imageio STATIC cli/image_io.cpp)
  target_include_directories(bc_imageio PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cli")
  target_include_directories(bc_imageio SYSTEM PRIVATE "${BC_THIRD_PARTY_INCLUDE}" "${BC_DEPS_DIR}/src/libwebp/src")
  target_link_libraries(bc_imageio PRIVATE webpdecoder)

  add_executable(beadcore-cli cli/main.cpp)
  target_link_libraries(beadcore-cli PRIVATE beadcore_internal bc_imageio)
  target_compile_definitions(beadcore-cli PRIVATE BC_REPO_ROOT="${BC_REPO_ROOT}")
  bc_copy_runtime(beadcore-cli)
endif()

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
if(BC_BUILD_TESTS)
  enable_testing()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory("${BC_DEPS_DIR}/src/googletest" "${CMAKE_BINARY_DIR}/_deps/googletest" EXCLUDE_FROM_ALL SYSTEM)

  file(GLOB BC_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
  add_executable(beadcore_tests ${BC_TEST_SOURCES})
  target_link_libraries(beadcore_tests PRIVATE beadcore_internal GTest::gtest_main)
  if(TARGET bc_imageio)
    target_link_libraries(beadcore_tests PRIVATE bc_imageio)
    target_compile_definitions(beadcore_tests PRIVATE BC_HAVE_IMAGEIO=1)
  endif()
  target_compile_definitions(beadcore_tests PRIVATE
    BC_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    BC_REPO_ROOT="${BC_REPO_ROOT}")
  bc_copy_runtime(beadcore_tests)
  include(GoogleTest)
  gtest_discover_tests(beadcore_tests DISCOVERY_TIMEOUT 60)
endif()
