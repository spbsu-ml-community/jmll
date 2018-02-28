package com.expleague.cuda;

import com.expleague.cuda.data.GPUMx;
import com.expleague.cuda.data.GPUVec;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;

/**
 * Created by hrundelb on 30.10.17.
 */
public class KernelOperations {

  private static final String CU_FILE_PATH = "array/Kernels.cu";

  private static final CUfunction F_MATRIXEXP =
      JCudaHelper.getFunction(CU_FILE_PATH, "fMatrixExp");

  public static void fMatrixExp(final GPUMx matrix, final GPUMx result) {
    final int rows = result.rows();

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(matrix.gpuVec.devicePointer),
        Pointer.to(result.gpuVec.devicePointer),
        Pointer.to(new int[]{rows})
    );

    callFunction(kernelParameters, F_MATRIXEXP, rows);
  }

  private static final CUfunction F_MATRIX_REDUCE =
      JCudaHelper.getFunction(CU_FILE_PATH, "fMatrixReduce");

  public static void fMatrixReduce(final GPUMx matrix, final GPUMx result) {

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(matrix.gpuVec.devicePointer),
        Pointer.to(result.gpuVec.devicePointer));

    final int blockDim = result.rows;

    JCudaDriver.cuLaunchKernel(F_MATRIX_REDUCE,
        blockDim, 1, 1,
        blockDim, 1, 1,
        0, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }


  private static final CUfunction REDUCE5 =
      JCudaHelper.getFunction(CU_FILE_PATH, "reduce5");

  public static void reduce5(GPUVec args, GPUVec result) {

    int dim = args.dim();
    final Pointer kernelParameters = Pointer.to(
        Pointer.to(args.devicePointer),
        Pointer.to(result.devicePointer),
        Pointer.to(new int[]{dim}));

    final int blockDim = 32;

    JCudaDriver.cuLaunchKernel(REDUCE5,
        dim / blockDim, 1, 1,
        blockDim, 1, 1,
        blockDim* Sizeof.FLOAT, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static final CUfunction F_FILL =
      JCudaHelper.getFunction(CU_FILE_PATH, "fFill");

  public static void dFill(final GPUVec vec, final float value) {
    final int length = vec.length;
    final Pointer kernelParameters = Pointer.to(
        Pointer.to(vec.devicePointer),
        Pointer.to(new float[]{value}),
        Pointer.to(new int[]{length}));

    final int blockDim = 16;
    JCudaDriver.cuLaunchKernel(F_FILL,
        length / blockDim + 1, 1, 1,
        blockDim, 1, 1,
        0, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static final CUfunction F_MATRIX_KERNEL_1 =
      JCudaHelper.getFunction(CU_FILE_PATH, "fMatrixKernel1");

  public static void fMatrixKernel1(final float weight, final float diff,
                                    final GPUVec distribution, final GPUVec expectedValue,
                                    final GPUMx betaGrad, final int to, final GPUMx weights) {

    final Pointer params = Pointer.to(
        Pointer.to(new int[]{betaGrad.rows()}),
        Pointer.to(new float[]{weight}),
        Pointer.to(new float[]{diff}),
        Pointer.to(distribution.devicePointer),
        Pointer.to(expectedValue.devicePointer),
        Pointer.to(betaGrad.gpuVec.devicePointer),
        Pointer.to(new int[]{to}),
        Pointer.to(weights.gpuVec.devicePointer)
    );

    final int len = betaGrad.rows() * betaGrad.columns();
    final int blockDim = 16;

    JCudaDriver.cuLaunchKernel(F_MATRIX_KERNEL_1,
        len / blockDim + 1, 1, 1,
        blockDim, 1, 1,
        0, null,
        params, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static final CUfunction F_MATRIX_KERNEL_2 =
      JCudaHelper.getFunction(CU_FILE_PATH, "fMatrixKernel2");

  public static void fMatrixKernel2(final float lambda, final GPUMx betaGrad, final int to,
                                    final GPUMx weights) {

    final Pointer params = Pointer.to(
        Pointer.to(new int[]{betaGrad.rows()}),
        Pointer.to(new float[]{lambda}),
        Pointer.to(betaGrad.gpuVec.devicePointer),
        Pointer.to(new int[]{to}),
        Pointer.to(weights.gpuVec.devicePointer)
    );

    final int len = betaGrad.rows() * betaGrad.columns();
    final int blockDim = 16;
    JCudaDriver.cuLaunchKernel(F_MATRIX_KERNEL_2,
        len / blockDim + 1, 1, 1,
        blockDim, 1, 1,
        0, null,
        params, null
    );
    JCudaDriver.cuCtxSynchronize();
  }

  private static final CUfunction F_VECTOR_KERNEL_1 =
      JCudaHelper.getFunction(CU_FILE_PATH, "fVectorKernel1");

  public static void fVectorKernel1(final GPUVec lastGrad, final GPUVec gradCoordinate,
                                    final GPUVec totalGrad, final float step,
                                    final int sumSize, final GPUVec result) {
    int len = result.dim();
    final Pointer params = Pointer.to(
        Pointer.to(lastGrad.devicePointer),
        Pointer.to(gradCoordinate.devicePointer),
        Pointer.to(totalGrad.devicePointer),
        Pointer.to(new float[]{step}),
        Pointer.to(new int[]{sumSize}),
        Pointer.to(result.devicePointer),
        Pointer.to(new int[]{len})
    );

    final int blockDim = 16;
    JCudaDriver.cuLaunchKernel(F_VECTOR_KERNEL_1,
        len / blockDim + 1, 1, 1,
        blockDim, 1, 1,
        0, null,
        params, null
    );
    JCudaDriver.cuCtxSynchronize();
  }


  private static void callFunction(final Pointer parameters, final CUfunction function, final int length) {
        int pow = upper2pow(length);
        int x = (int) Math.pow(pow, 1. / 3.);
        int z = x > 1024 ? 1024 : x;
        int y = pow / (z * x) + 1;

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
