package com.expleague.ml.clustering.impl;

import com.expleague.commons.math.metrics.Metric;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.clustering.ClusterizationAlgorithm;
import org.jetbrains.annotations.NotNull;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.function.Function;

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
  public Collection<? extends Collection<X>> cluster(final Collection<X> dataSet, final Function<X, Vec> data2DVector) {
    final Collection<Collection<X>> clusters = new HashSet<>();
    for (final X data : dataSet) {
      final Vec dataVector = data2DVector.apply(data);
      Collection<X> nearestCluster = null;
      double nearestDistance = Double.MAX_VALUE;
      double nearest2Distance = Double.MAX_VALUE;
      for (final Collection<X> cluster : clusters) {
        double minDistance = Double.MAX_VALUE;
        for (final X element : cluster) {
          final double candidateDistance = metric.distance(data2DVector.apply(element), dataVector);
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
        clusters.add(new HashSet<>(Collections.singletonList(data)));
      } else {
        nearestCluster.add(data);
      }
    }
    return clusters;
  }
}