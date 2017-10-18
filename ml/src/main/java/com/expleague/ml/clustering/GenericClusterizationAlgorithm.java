package com.expleague.ml.clustering;

import com.expleague.commons.func.Computable;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;

/**
 * User: terry
 * Date: 20.12.2009
 */
public interface GenericClusterizationAlgorithm<X, V> {
  @NotNull
  Collection<? extends Collection<X>> cluster(Collection<X> dataSet, Computable<X, V> data2DVector);
}
