package com.spbsu.exp.dl.functions.binary;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import com.spbsu.commons.util.ArrayPart;
import com.spbsu.exp.dl.functions.ArrayBinaryFunction;

/**
 * jmll
 * ksen
 * 10.December.2014 at 01:16
 */
public abstract class DoubleArrayBinaryFunction implements ArrayBinaryFunction<ArrayPart<double[]>, double[]> {

  @NotNull
  public ArrayVec f(final @NotNull ArrayVec x, final @NotNull ArrayVec z) {
    return new ArrayVec(f(x.data.array, z.data.array));
  }

  @NotNull
  public ColMajorArrayMx f(final @NotNull ColMajorArrayMx x, final @NotNull ColMajorArrayMx z) {
    return new ColMajorArrayMx(x.rows(), f(x.data, z.data));
  }

  @NotNull
  @Override
  public ArrayPart<double[]> f(final @NotNull ArrayPart<double[]> x, final @NotNull ArrayPart<double[]> z) {
    return new ArrayPart<>(f(x.array, z.array));
  }

  @NotNull
  @Override
  public double[] f(final @NotNull double[] x, final @NotNull double[] z) {
    final double[] valueBase = new double[x.length];

    map(x, z, valueBase);

    return valueBase;
  }

  @NotNull
  public ArrayVec df(final @NotNull ArrayVec x, final @NotNull ArrayVec z) {
    return new ArrayVec(df(x.data.array, z.data.array));
  }

  @NotNull
  public ColMajorArrayMx df(final @NotNull ColMajorArrayMx x, final @NotNull ColMajorArrayMx z) {
    return new ColMajorArrayMx(x.rows(), df(x.data, z.data));
  }

  @NotNull
  @Override
  public ArrayPart<double[]> df(final @NotNull ArrayPart<double[]> x, final @NotNull ArrayPart<double[]> z) {
    return new ArrayPart<>(df(x.array, z.array));
  }

  @NotNull
  @Override
  public double[] df(final @NotNull double[] x, final @NotNull double[] z) {
    final double[] valueBase = new double[x.length];

    dMap(x, z, valueBase);

    return valueBase;
  }

  protected abstract void map(final double[] x, final double[] z, final double[] y);

  protected abstract void dMap(final double[] x, final double[] z, final double[] y);

}
