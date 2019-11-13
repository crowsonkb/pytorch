#include <fbjni/fbjni.h>
#include <torch/csrc/api/include/torch/types.h>

#include "cmake_macros.h"

#if defined(TRACE_ENABLED) && defined(__ANDROID__)
#include <android/log.h>

#include <android/trace.h>
#include <dlfcn.h>

#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-jni", __VA_ARGS__)
#endif

namespace pytorch_jni {

class Trace {
 public:
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
  typedef void* (*fp_ATrace_beginSection)(const char* sectionName);
  typedef void* (*fp_ATrace_endSection)(void);

  static void* (*ATrace_beginSection)(const char* sectionName);
  static void* (*ATrace_endSection)(void);
#endif

  static void ensureInit();
  static void beginSection(const char* name);
  static void endSection();

  Trace(const char* name);
  ~Trace();

 private:
  static void init();
  static bool is_initialized_;
};

class JIValue : public facebook::jni::JavaClass<JIValue> {
public:
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/IValue;";

  constexpr static int kTypeCodeNull = 1;

  constexpr static int kTypeCodeTensor = 2;
  constexpr static int kTypeCodeBool = 3;
  constexpr static int kTypeCodeLong = 4;
  constexpr static int kTypeCodeDouble = 5;
  constexpr static int kTypeCodeString = 6;

  constexpr static int kTypeCodeTuple = 7;
  constexpr static int kTypeCodeBoolList = 8;
  constexpr static int kTypeCodeLongList = 9;
  constexpr static int kTypeCodeDoubleList = 10;
  constexpr static int kTypeCodeTensorList = 11;
  constexpr static int kTypeCodeList = 12;

  constexpr static int kTypeCodeDictStringKey = 13;
  constexpr static int kTypeCodeDictLongKey = 14;

  static facebook::jni::local_ref<JIValue> newJIValueFromAtIValue(
      const at::IValue& ivalue);

  static at::IValue JIValueToAtIValue(
      facebook::jni::alias_ref<JIValue> jivalue);
};
} // namespace pytorch_jni
