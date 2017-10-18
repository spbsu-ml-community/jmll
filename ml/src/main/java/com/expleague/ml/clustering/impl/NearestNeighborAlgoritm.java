package com.expleague.ml.clustering.impl;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.metrics.Metric;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.Factories;
import com.expleague.ml.clustering.ClusterizationAlgorithm;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;

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
  public Collection<? extends Collection<X>> cluster(final Collection<X> dataSet, final Computable<X, Vec> data2DVector) {
    final Collection<Collection<X>> clusters = Factories.hashSet();
    for (final X data : dataSet) {
      final Vec dataVector = data2DVector.compute(data);
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
