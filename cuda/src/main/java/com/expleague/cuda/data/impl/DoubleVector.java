package com.expleague.cuda.data.impl;

import com.expleague.cuda.JCudaMemory;
import com.expleague.cuda.data.ArrayBased;
import jcuda.driver.CUdeviceptr;
import org.jetbrains.annotations.NotNull;

/**
 * Created by hrundelb on 20.08.17.
 */
public class DoubleVector implements ArrayBased<double[]> {

  public int length;

  public CUdeviceptr devicePointer;

  public DoubleVector(final @NotNull double[] hostArray) {
    length = hostArray.length;
    devicePointer = JCudaMemory.alloCopy(hostArray);
  }

  @NotNull
  @Override
  public CUdeviceptr reproduce() {
    return JCudaMemory.allocDouble(length);
  }

  @NotNull
  @Override
  public DoubleVector set(@NotNull double[] hostArray) {
    JCudaMemory.copy(hostArray, devicePointer);
    return this;
  }

  @NotNull
  @Override
  public DoubleVector reset(@NotNull double[] hostArray) {
    JCudaMemory.destroy(devicePointer);
    length = hostArray.length;
    devicePointer = JCudaMemory.alloCopy(hostArray);
    return this;
  }

  @NotNull
  @Override
  public double[] get() {
    return JCudaMemory.copyDouble(devicePointer, length);
  }

  @Override
  public void setPointer(@NotNull CUdeviceptr devicePointer) {
    this.devicePointer = devicePointer;
  }

  @NotNull
  @Override
  public CUdeviceptr getPointer() {
    return devicePointer;
  }

  @Override
  public long length() {
    return length;
  }

  @Override
  public void destroy() {
    JCudaMemory.destroy(devicePointer);
  }
}
