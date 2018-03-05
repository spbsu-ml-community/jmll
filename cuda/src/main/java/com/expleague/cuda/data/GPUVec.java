package com.expleague.cuda.data;

import com.expleague.commons.math.vectors.OperableVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.cuda.DeviceOperations;
import com.expleague.cuda.JCudaMemory;
import com.expleague.cuda.KernelOperations;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

/**
 * Created by hrundelb on 27.08.17.
 */
public class GPUVec extends Vec.Stub implements OperableVec<GPUVec> {

  public int length;

  public CUdeviceptr devicePointer;

  public GPUVec(double[] hostArray) {
    this(convert(hostArray));
  }

  public GPUVec(float[] hostArray) {
    length = hostArray.length;
    devicePointer = JCudaMemory.alloCopy(hostArray);
  }

  public GPUVec(int length) {
    this(new float[length]);
  }

  private GPUVec(int length, CUdeviceptr devicePointer) {
    this.length = length;
    this.devicePointer = devicePointer;
  }

  public float[] toFloatArray() {
    return JCudaMemory.copy(devicePointer, length);
  }

  @Override
  public double[] toArray() {
    return convert(JCudaMemory.copy(devicePointer, length));
  }
//
//
//  public gpuVec set(double[] hostArray) {
//    if(length != hostArray.length) {
//      throw new IllegalArgumentException("length is not equal to hostArray.length");
//    }
//    devicePointer = JCudaMemory.copy(hostArray, devicePointer);
//    return this;
//  }


  @Override
  public double get(int i) {
    throw new UnsupportedOperationException();
    //return toArray()[i];
  }

  @Override
  public Vec set(int i, double val) {
    throw new UnsupportedOperationException();
//    double[] doubles = toArray();
//    doubles[i] = val;
//    return set(doubles);
  }

  @Override
  public Vec adjust(int i, double increment) {
    throw new UnsupportedOperationException();
  }

  @Override
  public VecIterator nonZeroes() {
    throw new UnsupportedOperationException();
  }

  @Override
  public int dim() {
    return length;
  }

  @Override
  public Vec sub(int start, int len) {
    CUdeviceptr cUdeviceptr = devicePointer.withByteOffset(start * Sizeof.FLOAT);
    return new GPUVec(len, cUdeviceptr);
  }

  public static float[] convert(double[] doubles) {
    float[] floats = new float[doubles.length];
    for (int i = 0; i < doubles.length; i++) {
      floats[i] = (float) doubles[i];
    }
    return floats;
  }

  public static double[] convert(float[] floats) {
    double[] doubles = new double[floats.length];
    for (int i = 0; i < doubles.length; i++) {
      doubles[i] = floats[i];
    }
    return doubles;
  }

  public void add(GPUVec other) {
    DeviceOperations.append(this, other);
  }

  public double mul(GPUVec other) {
    return DeviceOperations.multiply(this, other);
  }

  public void fill(double val) {
    KernelOperations.dFill(this, (float) val);
  }

  public void inscale(GPUVec other, double scale) {
    DeviceOperations.incscale(this, other, (float) scale);
  }

  public void scale(double scale) {
    DeviceOperations.scale(this, (float) scale);
  }
}
