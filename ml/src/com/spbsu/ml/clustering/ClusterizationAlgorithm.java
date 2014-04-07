package com.spbsu.ml.clustering;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;

/**
 * User: terry
 * Date: 20.12.2009
 */
public interface ClusterizationAlgorithm<X> {
  @NotNull
  Collection<? extends Collection<X>> cluster(Collection<X> dataSet, Computable<X, Vec> data2DVector);
}
