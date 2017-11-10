package com.expleague.ml.clustering.impl;

import com.expleague.commons.math.metrics.Metric;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.clustering.ClusterizationAlgorithm;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.function.Function;

/**
 * User: terry
 * Date: 16.01.2010
 */
public class NearestNeighborAlgoritm<X> implements ClusterizationAlgorithm<X> {
  private final Metric<Vec> metric;
  private final double acceptanceDistance;
  private final double rejectionDistance;

  public NearestNeighborAlgoritm(final Metric<Vec> metric, final double acceptanceDistance, final double rejectionDistance) {
    this.metric = metric;
    this.acceptanceDistance = acceptanceDistance;
    this.rejectionDistance = rejectionDistance;
  }

  @NotNull
  @Override
  public Collection<? extends Collection<X>> cluster(final Collection<X> dataSet, final Function<X, Vec> data2DVector) {
    final Collection<Collection<X>> clusters = new HashSet<>();
    for (final X data : dataSet) {
      final Vec dataVector = data2DVector.apply(data);
      Collection<X> nearestCluster = null;
      double nearestDistance = Double.MAX_VALUE;
      for (final Collection<X> cluster : clusters) {
        for (final X element : cluster) {
          final double candidateDistance = metric.distance(data2DVector.apply(element), dataVector);
          if (candidateDistance < nearestDistance && candidateDistance < acceptanceDistance) {
            nearestDistance = candidateDistance;
            nearestCluster = cluster;
          } else if (candidateDistance > rejectionDistance) break;
        }
      }
      if (nearestCluster == null) {
        clusters.add(new HashSet<>(Collections.singleton(data)));
      } else {
        nearestCluster.add(data);
      }
    }
    return clusters;
  }
}
