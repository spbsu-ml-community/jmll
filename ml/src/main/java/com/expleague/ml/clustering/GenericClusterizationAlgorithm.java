package com.expleague.ml.clustering;

import org.jetbrains.annotations.NotNull;

import java.util.Collection;
import java.util.function.Function;

/**
 * User: terry
 * Date: 20.12.2009
 */
public interface GenericClusterizationAlgorithm<X, V> {
  @NotNull
  Collection<? extends Collection<X>> cluster(Collection<X> dataSet, Function<X, V> data2DVector);
}
