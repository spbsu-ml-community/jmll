package com.spbsu.ml.cuda.data.impl;

import com.spbsu.ml.cuda.data.FVector;
import org.jetbrains.annotations.NotNull;
import gnu.trove.list.TIntList;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:56
 */
public class FArrayVector implements FVector {

  private float[] data;

  public FArrayVector(final float[] data) {
    this.data = data;
  }

  public FArrayVector(final int dimension) {
    data = new float[dimension];
  }

  @NotNull
  @Override
  public FVector reproduce(final @NotNull float[] base) {
    return new FArrayVector(base);
  }

  @Override
  public float get(final int index) {
    return data[index];
  }

  @NotNull
  @Override
  public FVector set(int index, float value) {
    data[index] = value;
    return this;
  }

  @NotNull
  @Override
  public FVector getRange(final @NotNull TIntList indexes) {
    final int size = indexes.size();
    final float[] subVector = new float[size];
    for (int i = 0; i < size; i++) {
      subVector[i] = data[indexes.get(i)];
    }
    return new FArrayVector(subVector);
  }

  @NotNull
  @Override
  public float[] toArray() {
    return data;
  }

  @Override
  public int getDimension() {
    return data.length;
  }

  public String toString() {
    return "[" + getDimension() + "]";
  }

  public String toString(int a) {
    final StringBuilder builder = new StringBuilder();
    for (int i = 0; i < data.length; i++) {
      builder.append(data[i]).append(' ');
    }
    return builder.append('\n').toString();
  }

}
