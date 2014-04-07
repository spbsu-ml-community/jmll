package com.spbsu.ml.clustering.impl;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.metrics.Metric;
import com.spbsu.ml.clustering.GenericClusterizationAlgorithm;
import com.spbsu.commons.util.Factories;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;

/**
 * User: terry
 * Date: 16.01.2010
 */
public class GenericNearestNeighborAlgoritm<X,V> implements GenericClusterizationAlgorithm<X, V> {
  private Metric<V> metric;
  private double acceptanceDistance;
  private double rejectionDistance;

  public GenericNearestNeighborAlgoritm(Metric<V> metric, double acceptanceDistance, double rejectionDistance) {
    this.metric = metric;
    this.acceptanceDistance = acceptanceDistance;
    this.rejectionDistance = rejectionDistance;
  }

  @NotNull
  @Override
  public Collection<? extends Collection<X>> cluster(Collection<X> dataSet, Computable<X, V> data2DVector) {
    final Collection<Collection<X>> clusters = Factories.hashSet();
    for (final X data : dataSet) {
      final V dataVector = data2DVector.compute(data);
      Collection<X> nearestCluster = null;
      double nearestDistance = Double.MAX_VALUE;
      for (final Collection<X> cluster : clusters) {
        for (final X element : cluster) {
          final double candidateDistance = metric.distance(data2DVector.compute(element), dataVector);
          if (candidateDistance < nearestDistance && candidateDistance < acceptanceDistance) {
            nearestDistance = candidateDistance;
            nearestCluster = cluster;
          } else if (candidateDistance > rejectionDistance) break;
        }
      }
      if (nearestCluster == null) {
        clusters.add(Factories.hashSet(data));
      } else {
        nearestCluster.add(data);
      }
    }
    return clusters;
  }
}
