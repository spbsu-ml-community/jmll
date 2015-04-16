package com.spbsu.exp.dl.functions;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.util.ArrayPart;

/**
 * jmll
 * ksen
 * 10.December.2014 at 01:08
 */
public interface ArrayBinaryFunction<T extends ArrayPart<R>, R> {

  @NotNull T f(final @NotNull T x, final @NotNull T z);

  @NotNull R f(final @NotNull R x, final @NotNull R z);

  @NotNull T df(final @NotNull T x, final @NotNull T z);

  @NotNull R df(final @NotNull R x, final @NotNull R z);

}
