package com.spbsu.exp.dl.utils;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import gnu.trove.list.array.TIntArrayList;

import java.util.Random;

/**
 * jmll
 * ksen
 * 25.November.2014 at 20:43
 */
public class DataUtils {

  public static double rmse(final @NotNull ColMajorArrayMx A, final @NotNull ColMajorArrayMx B) {
    return rmse(A.toArray(), B.toArray());
  }

  private static double rmse(final double[] a, final double[] b) {
    return Math.sqrt(mse(a, b));
  }

  private static double mse(final double[] a, final double[] b) {
    double sum = 0;
    final int length = a.length;
    for (int i = 0; i < length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return sum / length;
  }

  @NotNull
  public static ColMajorArrayMx repeatAsColumns(final @NotNull ArrayVec a, final int times) {
    return repeatAsColumns(a.toArray(), times);
  }

  @NotNull
  public static ColMajorArrayMx repeatAsColumns(final @NotNull double[] a, final int times) {
    return new ColMajorArrayMx(a.length, repeat(a, times));
  }

  private static double[] repeat(final double[] source, final int times) {
    final int length = source.length;
    final double[] destination = new double[times * length];

    for (int i = 0; i < times; i++) {
      System.arraycopy(source, 0, destination, i * length, length);
    }
    return destination;
  }

  @NotNull
  public static ArrayVec extendAsBottom(final @NotNull ArrayVec a, final double alpha) {
    final double[] aData = a.toArray();
    final double[] bData = new double[aData.length + 1];
    System.arraycopy(aData, 0, bData, 0, aData.length);
    bData[aData.length] = alpha;

    return new ArrayVec(bData);
  }

  @NotNull
  public static ArrayVec contractBottom(final @NotNull ArrayVec a) {
    final double[] aData = a.toArray();
    final double[] bData = new double[aData.length - 1];
    System.arraycopy(aData, 0, bData, 0, aData.length - 1);

    return new ArrayVec(bData);
  }

  @NotNull
  public static ColMajorArrayMx extendAsBottomRow(final @NotNull ColMajorArrayMx A, final @NotNull ArrayVec b) {
    final int rows = A.rows();
    final int columns = A.columns();
    final ColMajorArrayMx extended = new ColMajorArrayMx(rows + 1, columns);
    for (int i = 0; i < columns; i++) {
      extended.setPieceOfColumn(i, 0, A.col(i));
      extended.set(rows, i, b.get(i));
    }
    return extended;
  }

  @NotNull
  public static ColMajorArrayMx contractBottomRow(final @NotNull ColMajorArrayMx A) {
    final int rows = A.rows() - 1;
    final int columns = A.columns();
    final ColMajorArrayMx contracted = new ColMajorArrayMx(rows, columns);
    for (int i = 0; i < columns; i++) {
      contracted.setPieceOfColumn(i, 0, rows, A.col(i));
    }
    return contracted;
  }

  @NotNull
  public static ArrayVec once(final int size) {
    final double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = 1.;
    }
    return new ArrayVec(data);
  }

  @NotNull
  public static ColMajorArrayMx once(final int rows, final int columns) {
    final int dim = rows * columns;
    final double[] data = new double[dim];
    for (int i = 0; i < dim; i++) {
      data[i] = 1.;
    }
    return new ColMajorArrayMx(rows, data);
  }

  @NotNull
  public static double[] doubleOnce(final int size) {
    final double[] data = new double[size];
    for (int i = 0; i < size; i++) {
      data[i] = 1.;
    }
    return data;
  }

  public static TIntArrayList randomPermutations(final int size) {
    final Random random = new Random();
    final TIntArrayList list = new TIntArrayList(size);

    for (int i = 0; i < size; i++) {
      list.add(i);
    }
    list.shuffle(random);

    return list;
  }

}