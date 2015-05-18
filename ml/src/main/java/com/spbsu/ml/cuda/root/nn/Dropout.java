package com.spbsu.ml.cuda.root.nn;

import org.jetbrains.annotations.NotNull;

import com.spbsu.ml.cuda.JCudaHelper;
import com.spbsu.ml.cuda.JCurandHelper;
import com.spbsu.ml.cuda.data.impl.FloatVector;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcurand.curandGenerator;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class Dropout {

  static {
    JCudaHelper.hook();
  }


  private static final String CU_FILE_PATH = "nn/Dropout.cu";

  private static final CUfunction DROPOUT_TRAIN = JCudaHelper.getFunction(CU_FILE_PATH, "dropoutTrain");

  public static void dropoutTrain(
      final @NotNull FloatVector input,
      final @NotNull FloatVector dropoutMask,
      final @NotNull FloatVector output,
      final @NotNull curandGenerator generator,
      final float dropoutFraction
  ) {
    final int length = input.length;
    JCurandHelper.generateUniform(dropoutMask.devicePointer, length, generator);

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(dropoutMask.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new float[]{dropoutFraction}),
        Pointer.to(new long[]{length})
    );

    final int pow = upper2pow(length);
    final int x = (int) Math.pow(pow, 1. / 3.);
    final int z = x > 1024 ? 1024 : x;
    final int y = pow / (z * x);

    JCudaDriver.cuLaunchKernel(DROPOUT_TRAIN,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static final CUfunction DROPOUT_TEST = JCudaHelper.getFunction(CU_FILE_PATH, "dropoutTest");

  public static void dropoutTest(
      final @NotNull FloatVector input,
      final @NotNull FloatVector output,
      final float dropoutFraction
  ) {
    final int length = input.length;

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(input.devicePointer),
        Pointer.to(output.devicePointer),
        Pointer.to(new float[]{dropoutFraction}),
        Pointer.to(new long[]{length})
    );

    final int pow = upper2pow(length);
    final int x = (int) Math.pow(pow, 1. / 3.);
    final int z = x > 1024 ? 1024 : x;
    final int y = pow / (z * x);

    JCudaDriver.cuLaunchKernel(DROPOUT_TEST,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static int upper2pow(final int value) {
    return (int) Math.pow(2, 32 - Integer.numberOfLeadingZeros(value - 1));
  }

}
