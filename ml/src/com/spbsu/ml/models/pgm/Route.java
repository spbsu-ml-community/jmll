package com.spbsu.ml.models.pgm;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.iterators.SimpleVecIterator;

/**
 * User: solar
 * Date: 07.04.14
 * Time: 21:36
 */
public abstract class Route implements Vec{
  @Override
  public double get(int i) {
    return dst(i) + 1;
  }

  @Override
  public Vec set(int i, double val) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Vec adjust(int i, double increment) {
    throw new UnsupportedOperationException();
  }

  @Override
  public VecIterator nonZeroes() {
    return new SimpleVecIterator(this);
  }

  @Override
  public int dim() {
    return length();
  }

  @Override
  public double[] toArray() {
    final double[] array = new double[length()];
    for (int i = 0; i < array.length; i++) {
      array[i] = get(i);
    }
    return array;
  }

  @Override
  public Vec sub(int start, int len) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder(20);
    builder.append('(');
    for (int i = 0; i < length(); i++) {
      if (i > 0)
        builder.append(',');
      builder.append(dst(i));
    }
    builder.append(')').append("->").append(p());
    return builder.toString();
  }

  public abstract double p();
  public abstract int last();
  public abstract int length();

  public abstract ProbabilisticGraphicalModel dstOwner(int stepNo);
  public abstract int dst(int stepNo);
}
