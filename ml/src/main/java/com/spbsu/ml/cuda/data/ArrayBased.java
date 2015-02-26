package com.spbsu.ml.cuda.data;

import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:09
 */
public interface ArrayBased<T> {

  @NotNull T toArray();

  @NotNull ArrayBased<T> reproduce(final @NotNull T base);

}
