package com.expleague.ml.clustering.impl;

import com.expleague.commons.math.metrics.Metric;
import com.expleague.ml.clustering.ClusterizationAlgorithm;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.util.CollectionTools;
import com.expleague.commons.util.Pair;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.function.Function;

/**
 * User: terry
 * Date: 16.01.2010
 */
public class NearestCentroidDRAlgorithm<X> implements ClusterizationAlgorithm<X> {
  private final Metric<Vec> metric;
  private final double acceptanceDistance;
  private final double distanceRatio;

  public NearestCentroidDRAlgorithm(final Metric<Vec> metric, final double acceptanceDistance, final double distanceRatio) {
    this.metric = metric;
    this.acceptanceDistance = acceptanceDistance;
    this.distanceRatio = distanceRatio;
  }

  @NotNull
  @Override
  public Collection<? extends Collection<X>> cluster(final Collection<X> dataSet, final Function<X, Vec> data2DVector) {
    final Collection<Pair<Collection<X>,Vec>> clusters = new HashSet<>();
    for (final X data : dataSet) {
      final Vec dataVector = data2DVector.apply(data);
      Pair<Collection<X>, Vec> nearestCluster = null;
      double nearestDistance = Double.MAX_VALUE;
      double nearest2Distance = Double.MAX_VALUE;
      for (final Pair<Collection<X>, Vec> pair : clusters) {
        final double candidateDistance = metric.distance(pair.getSecond(), dataVector);
        if (candidateDistance < nearestDistance) {
          nearestDistance = candidateDistance;
          nearestCluster = pair;
        }
        else if (candidateDistance < nearest2Distance) {
          nearest2Distance = candidateDistance;
        }
      }
      final boolean good = nearestDistance < acceptanceDistance && (nearest2Distance == Double.MAX_VALUE || nearestDistance / nearest2Distance < distanceRatio);
      if (nearestCluster == null || !good) {
        clusters.add(Pair.<Collection<X>,Vec>create(new HashSet<>(Collections.singleton(data)), dataVector));
      } else {
        final Collection<X> collection = nearestCluster.getFirst();
        final Vec centroid = nearestCluster.getSecond();
        VecTools.scale(centroid, collection.size());
        VecTools.append(centroid, dataVector);
        collection.add(data);
        VecTools.scale(centroid, 1./collection.size());
      }
    }
    return CollectionTools.mapFirst(clusters);
  }
}