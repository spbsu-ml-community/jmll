package com.spbsu.ml.cuda.data;

import org.jetbrains.annotations.NotNull;
import gnu.trove.list.TIntList;

/**
 * jmll
 * ksen
 * 22.October.2014 at 22:46
 */
public interface FMatrix extends ArrayBased<float[]> {

  FMatrix set(final int i, final int j, final float value);

  float get(final int i, final int j);

  @NotNull FVector getColumn(final int j);

  @NotNull FMatrix getColumnsRange(final int begin, final int length);

  @NotNull FMatrix getColumnsRange(final @NotNull TIntList indexes);

  void setColumn(final int j, final @NotNull FVector column);

  void setColumn(final int j, final float[] column);

  void setPieceOfColumn(final int j, final int begin, final @NotNull FVector piece);

  void setPieceOfColumn(final int j, final int begin, final int length, final @NotNull FVector piece);

  int getRows();

  int getColumns();

}
