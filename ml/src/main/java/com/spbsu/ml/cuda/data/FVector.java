package com.spbsu.ml.cuda.data;

import gnu.trove.list.TIntList;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:55
 */
public interface FVector extends ArrayBased<float[]> {

  float get(final int index);

  @NotNull FVector set(final int index, final float value);

  @NotNull FVector getRange(final @NotNull TIntList indexes);

  int getDimension();

}
