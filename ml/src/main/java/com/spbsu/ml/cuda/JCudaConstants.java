package com.spbsu.ml.cuda;

/**
 * jmll
 * ksen
 * 21.February.2015 at 13:53
 */
public class JCudaConstants {

  public static final String[] JCUDA_NATIVE_LIBS_NAMES = {
      "JCudaDriver-linux-x86_64.so",
      "JCudaRuntime-linux-x86_64.so",
      "JCublas-linux-x86_64.so",
      "JCublas2-linux-x86_64.so",
      "JCurand-linux-x86_64.so",
      "JCusparse-linux-x86_64.so",
      "JCufft-linux-x86_64.so"
  };

  public static final String JCUDA_TMP_DIRECTORY_NAME = "jcuda_working_directory";

  public static final String JCUDA_CLASS_PATH = "/com/spbsu/ml/cuda/";
  public static final String CU_CLASS_PATH = JCUDA_CLASS_PATH + "cu/";
  public static final String PTX_CLASS_PATH = JCUDA_CLASS_PATH + "ptx/";

}
