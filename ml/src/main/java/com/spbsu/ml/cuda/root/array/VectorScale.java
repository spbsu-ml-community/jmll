package com.spbsu.ml.cuda.root.array;

import org.jetbrains.annotations.NotNull;

import com.spbsu.ml.cuda.JCudaHelper;
import com.spbsu.ml.cuda.data.impl.FloatVector;
import jcuda.Pointer;
import jcuda.driver.*;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class VectorScale {

  static {
    JCudaHelper.hook();
  }


  private static final String CU_FILE_PATH = "array/VectorScale.cu";

  private static final CUfunction F_SIGMOID = JCudaHelper.getFunction(CU_FILE_PATH, "fSigmoid");

  public static void fSigmoid(
      final @NotNull FloatVector input,
      final @NotNull FloatVector output
  ) {
    final int length = input.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new int[]{length})
    );
    callFunction(kernelParameters, F_SIGMOID, length);
  }

  private static final CUfunction F_DER_SIGMOID = JCudaHelper.getFunction(CU_FILE_PATH, "fDerSigmoid");

  public static void fDerSigmoid(
      final @NotNull FloatVector input,
      final @NotNull FloatVector output
  ) {
    final int length = input.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new int[]{length})
    );
    callFunction(kernelParameters, F_DER_SIGMOID, length);
  }

  private static final CUfunction F_EXP = JCudaHelper.getFunction(CU_FILE_PATH, "fExp");

  public static void fExp(final @NotNull FloatVector input, final @NotNull FloatVector output) {
    final int length = input.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new long[]{length})
    );
    callFunction(kernelParameters, F_EXP, length);
  }

  private static final CUfunction F_TANH = JCudaHelper.getFunction(CU_FILE_PATH, "fTanh");

  public static void fTanh(final @NotNull FloatVector input, final @NotNull FloatVector output) {
    final int length = input.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new long[]{length})
    );
    callFunction(kernelParameters, F_TANH, length);
  }

  private static final CUfunction F_NEGATION = JCudaHelper.getFunction(CU_FILE_PATH, "fNegation");

  public static void fNegation(final @NotNull FloatVector input, final @NotNull FloatVector output) {
    final int length = input.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new long[]{length})
    );
    callFunction(kernelParameters, F_NEGATION, length);
  }

  private static final CUfunction F_HADAMARD = JCudaHelper.getFunction(CU_FILE_PATH, "fHadamard");

  public static void fHadamard(
      final @NotNull FloatVector left,
      final @NotNull FloatVector right,
      final @NotNull FloatVector output
  ) {
    final int length = output.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(left.devicePointer),
        Pointer.to(right.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new long[]{length})
    );
    callFunction(kernelParameters, F_HADAMARD, length);
  }

  private static void callFunction(final Pointer parameters, final CUfunction function, final int length) {
    int pow = upper2pow(length);
    int x = (int) Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);

    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        parameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static int upper2pow(final int value) {
    return (int) Math.pow(2, 32 - Integer.numberOfLeadingZeros(value - 1));
  }

}
