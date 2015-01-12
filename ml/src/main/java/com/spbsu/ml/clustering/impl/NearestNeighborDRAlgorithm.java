package com.spbsu.ml.clustering.impl;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.metrics.Metric;
import com.spbsu.ml.clustering.ClusterizationAlgorithm;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Factories;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;

/**
 * User: terry
 * Date: 16.01.2010
 */
public class NearestNeighborDRAlgorithm<X> implements ClusterizationAlgorithm<X> {
  private final Metric<Vec> metric;
  private final double acceptanceDistance;
  private final double distanceRatio;

  public NearestNeighborDRAlgorithm(final Metric<Vec> metric, final double acceptanceDistance, final double distanceRatio) {
    this.metric = metric;
    this.acceptanceDistance = acceptanceDistance;
    this.distanceRatio = distanceRatio;
  }

  @NotNull
  @Override
  public Collection<? extends Collection<X>> cluster(final Collection<X> dataSet, final Computable<X, Vec> data2DVector) {
    final Collection<Collection<X>> clusters = Factories.hashSet();
    for (final X data : dataSet) {
      final Vec dataVector = data2DVector.compute(data);
      Collection<X> nearestCluster = null;
      double nearestDistance = Double.MAX_VALUE;
      double nearest2Distance = Double.MAX_VALUE;
      for (final Collection<X> cluster : clusters) {
        double minDistance = Double.MAX_VALUE;
        for (final X element : cluster) {
          final double candidateDistance = metric.distance(data2DVector.compute(element), dataVector);
          minDistance = Math.min(minDistance, candidateDistance);
        }

        if (minDistance < nearestDistance) {
          nearestDistance = minDistance;
          nearestCluster = cluster;
        }
        else if (minDistance < nearest2Distance) {
          nearest2Distance = minDistance;
        }
      }

      final boolean good =
        (nearestDistance < acceptanceDistance && (nearest2Distance == Double.MAX_VALUE || nearestDistance / nearest2Distance < distanceRatio));
      if (nearestCluster == null || !good) {
        clusters.add(Factories.hashSet(data));
      } else {
        nearestCluster.add(data);
      }
    }
    return clusters;
  }
}