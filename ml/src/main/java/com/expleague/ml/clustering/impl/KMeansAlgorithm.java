package com.expleague.ml.clustering.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.clustering.ClusterizationAlgorithm;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.function.Function;

import static com.expleague.commons.math.vectors.VecTools.scale;

/**
 * User: solar
 * Date: 13.02.2010
 * Time: 22:49:47
 */
public class KMeansAlgorithm<T> implements ClusterizationAlgorithm<T> {
  int clustCount;
  double maxDist;

  public KMeansAlgorithm(final int clustCount, final double maxDist) {
    this.clustCount = clustCount;
    this.maxDist = maxDist;
  }

  @NotNull
  @Override
  public Collection<? extends Collection<T>> cluster(final Collection<T> dataSet, final Function<T, Vec> data2DVector) {
    Vec[] centroids = new Vec[clustCount];
    final List<Set<T>> clusters = new ArrayList<Set<T>>();
    while (clusters.size() < centroids.length) {
      clusters.add(new HashSet<T>());
    }
    int fullIndex = 0;
    for (final T point : dataSet) {
      final Vec vec = data2DVector.apply(point);
      final int index = fullIndex++ % centroids.length;
      if (centroids[index] == null)
        //noinspection unchecked
        centroids[index] = VecTools.copy(vec);
      else
        VecTools.append(centroids[index], vec);
      clusters.get(index).add(point);
    }
    for (int i = 0; i < centroids.length; i++) {
      VecTools.scale(centroids[i], 1./clusters.size());
    }

    int iteration = 0;
    do {
      final Vec[] nextCentroids = new Vec[clustCount];
      for (int i = 0; i < centroids.length; i++) {
        clusters.get(i).clear();
      }

      for (final T point : dataSet) {
        final Vec vec = data2DVector.apply(point);
        double minResemblance = Double.MAX_VALUE;
        int minIndex = -1;
        for (int i = 0; i < centroids.length; i++) {
          final Vec centroid = centroids[i];
          final double resemblance = VecTools.distanceAV(centroid, vec);
          if (resemblance < minResemblance) {
            minResemblance = resemblance;
            minIndex = i;
          }
        }
        clusters.get(minIndex).add(point);
        VecTools.append(nextCentroids[minIndex], vec);
      }

      for (int i = 0; i < centroids.length; i++) {
        VecTools.scale(centroids[i], 1./clusters.size());
      }
      centroids = nextCentroids;
    }
    while (++iteration < 10);

    final Iterator<Set<T>> iter = clusters.iterator();
    int index = 0;
    while (iter.hasNext()) {
      final Set<T> cluster = iter.next();
      double meanDist = 0;
      final Vec centroid = centroids[index++];
      for (final T term : cluster) {
        meanDist += VecTools.distanceAV(data2DVector.apply(term), centroid);
      }
      meanDist /= cluster.size();
      if (meanDist > maxDist) {
//        iter.remove();
      }
    }

    return clusters;
  }
}
