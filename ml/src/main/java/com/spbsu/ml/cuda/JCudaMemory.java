package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

/**
 * jmll
 * ksen
 * 08.April.2015 at 00:20
 */
public class JCudaMemory {

  static {
    JCudaHelper.hook();
  }

  // DOUBLE

  public static CUdeviceptr alloCopy(final double[] data) {
    final int length = data.length;

    final CUdeviceptr devicePointer = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(devicePointer, length * Sizeof.DOUBLE);
    JCudaDriver.cuMemcpyHtoD(devicePointer, Pointer.to(data), length * Sizeof.DOUBLE);

    return devicePointer;
  }

  public static double[] copyDestr(final @NotNull CUdeviceptr devicePointer, final int length) {
    return copyDestr(devicePointer, new double[length]);
  }

  public static double[] copyDestr(final @NotNull CUdeviceptr devicePointer, final double[] hostData) {
    JCudaDriver.cuMemcpyDtoH(Pointer.to(hostData), devicePointer, hostData.length * Sizeof.DOUBLE);
    JCudaDriver.cuMemFree(devicePointer);

    return hostData;
  }

}
