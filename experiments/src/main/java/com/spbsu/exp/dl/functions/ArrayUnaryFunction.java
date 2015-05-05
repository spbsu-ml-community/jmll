package com.spbsu.exp.dl.functions;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.util.ArrayPart;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:14
 */
public interface ArrayUnaryFunction<T extends ArrayPart<R>, R> {

  @NotNull T f(final @NotNull T x);

  @NotNull R f(final @NotNull R x);

  @NotNull T df(final @NotNull T x);

  @NotNull R df(final @NotNull R x);

}
