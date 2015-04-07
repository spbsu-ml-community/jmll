package com.spbsu.ml.cuda;

import jcuda.Pointer;
import jcuda.driver.*;

/**
 * jmll
 * ksen
 * 25.October.2014 at 21:36
 */
public class JCudaVectorInscale { //todo(ksen): handle exit code

  static {
    JCudaHelper.hook();
  }

  public static void exp(final double[] ha) {
    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dExp");

    final CUdeviceptr da = JCudaMemory.alloCopy(ha);

    final int length = ha.length;
    final Pointer kernelParameters = Pointer.to(
        Pointer.to(da),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int) Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaMemory.copyDestr(da, ha);
  }

  public static void sigmoid(final double[] ha) {
    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dSigmoid");

    final CUdeviceptr da = JCudaMemory.alloCopy(ha);

    final int length = ha.length;
    final Pointer kernelParameters = Pointer.to(
        Pointer.to(da),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int) Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaMemory.copyDestr(da, ha);
  }

  public static void tanh(final double[] ha) {
    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dTanh");

    final CUdeviceptr da = JCudaMemory.alloCopy(ha);

    final int length = ha.length;
    final Pointer kernelParameters = Pointer.to(
        Pointer.to(da),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int) Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaMemory.copyDestr(da, ha);
  }

  private static int upper2pow(final int value) {
    return (int) Math.pow(2, 32 - Integer.numberOfLeadingZeros(value - 1));
  }

}
