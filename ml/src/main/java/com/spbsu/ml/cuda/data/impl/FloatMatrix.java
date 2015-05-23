package com.spbsu.ml.cuda.data.impl;

import org.jetbrains.annotations.NotNull;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class FloatMatrix extends FloatVector {

  public final int rows;
  public final int columns;

  public FloatMatrix(final int rows, final int columns) {
    this(rows, new float[rows * columns]);
  }

  public FloatMatrix(final int rows, final @NotNull float[] hostArray) {
    super(hostArray);
    this.rows = rows;
    this.columns = hostArray.length / rows;
  }

}
