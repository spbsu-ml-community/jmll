package com.expleague.cuda.data.impl;

import com.expleague.cuda.JCudaMemory;
import com.expleague.cuda.data.ArrayBased;
import org.jetbrains.annotations.NotNull;

import jcuda.driver.CUdeviceptr;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class FloatVector implements ArrayBased<float[]> {

  public int length;

  public CUdeviceptr devicePointer;

  public FloatVector(final @NotNull float[] hostArray) {
    length = hostArray.length;
    devicePointer = JCudaMemory.alloCopy(hostArray);
  }

  @NotNull
  @Override
  public CUdeviceptr reproduce() {
    return JCudaMemory.allocFloat(length);
  }

  @NotNull
  @Override
  public FloatVector set(final @NotNull float[] hostArray) {
    JCudaMemory.copy(hostArray, devicePointer);
    return this;
  }

  @NotNull
  @Override
  public FloatVector reset(final @NotNull float[] hostArray) {
    JCudaMemory.destroy(devicePointer);
    length = hostArray.length;
    devicePointer = JCudaMemory.alloCopy(hostArray);
    return this;
  }

  @NotNull
  @Override
  public float[] get() {
    return JCudaMemory.copy(devicePointer, length);
  }

  @Override
  public void setPointer(final @NotNull CUdeviceptr devicePointer) {
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
