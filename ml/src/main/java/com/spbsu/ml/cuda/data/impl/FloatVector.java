package com.spbsu.ml.cuda.data.impl;

import org.jetbrains.annotations.NotNull;

import com.spbsu.ml.cuda.JCudaMemory;
import jcuda.driver.CUdeviceptr;

import com.spbsu.ml.cuda.data.ArrayBased;

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
