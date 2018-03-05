package com.expleague.cuda.root.nn;

import com.expleague.cuda.JCudaHelper;
import com.expleague.cuda.data.GPUVec;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;

/**
 * Created by hrundelb on 28.01.18.
 */
public class NetworkOperations {

  private static final String CU_FILE_PATH = "nn/LayeredNetwork.cu";

  private static final int BLOCK_SIZE = 256;


  private static final CUfunction PRODUCE_STATE2 =
      JCudaHelper.getFunction(CU_FILE_PATH, "produceState2");

  public static void produceState2(GPUVec args, int argsSize, GPUVec weights,
                                   CUdeviceptr topology, int topSize, GPUVec outStates) {
    if (argsSize > BLOCK_SIZE)
      throw new IllegalArgumentException();


    int batchSize = args.dim() / argsSize;
    int dim = argsSize + topSize;
    //System.out.println(String.format("Batch: %s, ArgsSize: %s, TopSize: %s, Dim: %s",
    //    batchSize, argsSize, topSize, dim));

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(args.devicePointer),
        Pointer.to(new int[]{argsSize}),
        Pointer.to(weights.devicePointer),
        Pointer.to(topology),
        Pointer.to(new int[]{topSize}),
        Pointer.to(outStates.devicePointer));

    JCudaDriver.cuLaunchKernel(PRODUCE_STATE2,
        batchSize, 1, 1,
        BLOCK_SIZE, 1, 1,
        dim * Sizeof.FLOAT + dim * Sizeof.BYTE, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }


  private static final CUfunction PRODUCE_STATE3 =
      JCudaHelper.getFunction(CU_FILE_PATH, "produceState3");

  public static void produceState3(GPUVec args, int argsSize, GPUVec weights,
                                   CUdeviceptr topology, int topSize, GPUVec outStates) {
    if (argsSize > BLOCK_SIZE)
      throw new IllegalArgumentException();

    int batchSize = args.dim() / argsSize;
    int dim = argsSize + topSize;
    //System.out.println(String.format("Batch: %s, ArgsSize: %s, TopSize: %s, Dim: %s",
    //    batchSize, argsSize, topSize, dim));

    final Pointer kernelParameters = Pointer.to(
        Pointer.to(args.devicePointer),
        Pointer.to(new int[]{argsSize}),
        Pointer.to(weights.devicePointer),
        Pointer.to(topology),
        Pointer.to(new int[]{topSize}),
        Pointer.to(outStates.devicePointer));

    JCudaDriver.cuLaunchKernel(PRODUCE_STATE3,
        batchSize, 1, 1,
        BLOCK_SIZE, 1, 1,
        dim * Sizeof.FLOAT + BLOCK_SIZE * Sizeof.INT, null,
        kernelParameters, null
    );
    JCudaDriver.cuCtxSynchronize();
  }
}
