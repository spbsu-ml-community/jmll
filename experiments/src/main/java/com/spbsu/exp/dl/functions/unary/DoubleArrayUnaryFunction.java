package com.spbsu.exp.dl.functions.unary;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import com.spbsu.commons.util.ArrayPart;
import com.spbsu.exp.dl.functions.ArrayUnaryFunction;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:12
 */
public abstract class DoubleArrayUnaryFunction implements ArrayUnaryFunction<ArrayPart<double[]>, double[]> {

  @NotNull
  public ArrayVec f(final @NotNull ArrayVec x) {
    return new ArrayVec(f(x.data.array));
  }

  @NotNull
  public ColMajorArrayMx f(final @NotNull ColMajorArrayMx x) {
    return new ColMajorArrayMx(x.rows(), f(x.data));
  }

  @NotNull
  @Override
  public ArrayPart<double[]> f(final @NotNull ArrayPart<double[]> x) {
    return new ArrayPart<>(f(x.array));
  }

  @NotNull
  @Override
  public double[] f(final @NotNull double[] x) {
    final double[] valueBase = new double[x.length];

    map(x, valueBase);

    return valueBase;
  }

  @NotNull
  public ArrayVec df(final @NotNull ArrayVec x) {
    return new ArrayVec(df(x.data.array));
  }

  @NotNull
  public ColMajorArrayMx df(final @NotNull ColMajorArrayMx x) {
    return new ColMajorArrayMx(x.rows(), df(x.data));
  }

  @NotNull
  @Override
  public ArrayPart<double[]> df(final @NotNull ArrayPart<double[]> x) {
    return new ArrayPart<>(df(x.array));
  }

  @NotNull
  @Override
  public double[] df(final @NotNull double[] x) {
    final double[] valueBase = new double[x.length];

    dMap(x, valueBase);

    return valueBase;
  }

  protected abstract void map(final double[] x, final double[] y);

  protected abstract void dMap(final double[] x, final double[] y);

}
