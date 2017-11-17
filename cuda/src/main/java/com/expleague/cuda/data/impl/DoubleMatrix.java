package com.expleague.cuda.data.impl;

import org.jetbrains.annotations.NotNull;

/**
 * Created by hrundelb on 20.08.17.
 */
public class DoubleMatrix extends DoubleVector {
  public final int rows;
  public final int columns;

  public DoubleMatrix(final int rows, final int columns) {
    this(rows, new double[rows * columns]);
  }

  public DoubleMatrix(final int rows, final @NotNull double[] hostArray) {
    super(hostArray);
    this.rows = rows;
    this.columns = hostArray.length / rows;
  }
}
