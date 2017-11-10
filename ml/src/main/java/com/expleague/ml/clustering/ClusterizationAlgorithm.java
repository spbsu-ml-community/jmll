package com.expleague.ml.clustering;

import com.expleague.commons.math.vectors.Vec;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;
import java.util.function.Function;

/**
 * User: terry
 * Date: 20.12.2009
 */
public interface ClusterizationAlgorithm<X> {
  @NotNull
  Collection<? extends Collection<X>> cluster(Collection<X> dataSet, Function<X, Vec> data2DVector);
}
