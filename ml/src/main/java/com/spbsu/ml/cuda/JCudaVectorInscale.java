package com.spbsu.ml.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.jcurand.JCurand.*;
import static jcuda.runtime.JCuda.*;

/**
 * jmll
 * ksen
 * 25.October.2014 at 21:36
 */
public class JCudaVectorInscale {

  static {
    JCudaHelper.hook();
  }

  public static void exp(final double[] ha) {
    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dExp");

    final int length = ha.length;

    final CUdeviceptr da = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(da, length * Sizeof.DOUBLE);
    JCudaDriver.cuMemcpyHtoD(da, Pointer.to(ha), length * Sizeof.DOUBLE);

    Pointer kernelParameters = Pointer.to(
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

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), da, length * Sizeof.DOUBLE);

    JCudaDriver.cuMemFree(da);
  }

  public static void sigmoid(final double[] ha) {
    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dSigmoid");

    final int length = ha.length;

    final CUdeviceptr da = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(da, length * Sizeof.DOUBLE);
    JCudaDriver.cuMemcpyHtoD(da, Pointer.to(ha), length * Sizeof.DOUBLE);

    Pointer kernelParameters = Pointer.to(
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

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), da, length * Sizeof.DOUBLE);

    JCudaDriver.cuMemFree(da);
  }

  public static void tanh(final double[] ha) {
    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dTanh");

    final int length = ha.length;

    final CUdeviceptr da = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(da, length * Sizeof.DOUBLE);
    JCudaDriver.cuMemcpyHtoD(da, Pointer.to(ha), length * Sizeof.DOUBLE);

    Pointer kernelParameters = Pointer.to(
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

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), da, length * Sizeof.DOUBLE);

    JCudaDriver.cuMemFree(da);
  }

  private static void fRndSigmoid(final double[] ha) {
    final int length = ha.length;
    final double[] randomH = getRandom(length);

    JCudaDriver.setExceptionsEnabled(true);

    JCudaDriver.cuInit(0);
    final CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);

    final CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    final CUfunction function = JCudaHelper.getFunction("VectorInscale.cu", "dRndSigmoid");

    final CUdeviceptr original = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(original, length * Sizeof.DOUBLE);
    JCudaDriver.cuMemcpyHtoD(original, Pointer.to(ha), length * Sizeof.DOUBLE);

    final CUdeviceptr randomD = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(randomD, length * Sizeof.DOUBLE);
    JCudaDriver.cuMemcpyHtoD(randomD, Pointer.to(randomH), length * Sizeof.DOUBLE);

    Pointer kernelParameters = Pointer.to(
        Pointer.to(original),
        Pointer.to(randomD),
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

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), original, length * Sizeof.DOUBLE);

    JCudaDriver.cuMemFree(original);

    JCudaDriver.cuCtxDestroy(context);
  }

  private static double[] getRandom(final int size) {
    JCuda.setExceptionsEnabled(true);
    JCurand.setExceptionsEnabled(true);

    final curandGenerator generator = new curandGenerator();

    double host[] = new double[size];
    final Pointer device = new Pointer();
    cudaMalloc(device, size * Sizeof.DOUBLE);

    curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis());

    curandGenerateUniform(generator, device, size);

    cudaMemcpy(Pointer.to(host), device, size * Sizeof.DOUBLE, cudaMemcpyKind.cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);
    cudaFree(device);

    return host;
  }

  private static int upper2pow(final int value) {
    return (int) Math.pow(2, 32 - Integer.numberOfLeadingZeros(value - 1));
  }

}
