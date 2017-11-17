package com.expleague.cuda.data;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;

/**
 * Created by hrundelb on 06.09.17.
 */
public class GPUMx extends Mx.Stub {

  public final GPUVec gpuVec;
  public final int rows;

  public GPUMx(int rows, Vec gpuVec) {
    this.gpuVec = (GPUVec) gpuVec;
    this.rows = rows;
  }

  public GPUMx(int rows, int columns) {
    this.gpuVec = new GPUVec(new float[rows * columns]);
    this.rows = rows;
  }

  public float[] toFloatArray() {
    return gpuVec.toFloatArray();
  }

  @Override
  public double get(int i) {
    return gpuVec.get(i);
  }

  @Override
  public Vec set(int i, double val) {
    return gpuVec.set(i, val);
  }

  @Override
  public Vec adjust(int i, double increment) {
    return gpuVec.adjust(i, increment);
  }

  @Override
  public double get(int i, int j) {
    return gpuVec.get(i * columns() + j);
  }

  @Override
  public Mx set(int i, int j, double val) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Mx adjust(int i, int j, double increment) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Mx sub(int i, int j, int height, int width) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int columns() {
    return gpuVec.dim() / rows;
  }

  @Override
  public int rows() {
    return rows;
  }

  @Override
  public double[] toArray() {
    return gpuVec.toArray();
  }
}
