package com.expleague.ml.clustering.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.clustering.ClusterizationAlgorithm;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.util.*;
import java.util.function.Function;

import static com.expleague.commons.math.vectors.VecTools.scale;

/**
 * User: solar
 * Date: 13.02.2010
 * Time: 20:32:44
 */
public class ForelAlgorithm<T> implements ClusterizationAlgorithm<T> {
  private final double maxDist0;

  public ForelAlgorithm(final double maxDist0) {
    this.maxDist0 = maxDist0;
  }//  @Required

  @NotNull
  @Override
  public Collection<? extends Collection<T>> cluster(final Collection<T> dataSet, final Function<T, Vec> data2DVector) {
    int count = 0;
    final List<Set<T>> clusters = new ArrayList<Set<T>>();
    final Set<T> unclassified = new HashSet<T>(dataSet);
    while (!unclassified.isEmpty()) {
      final T first = unclassified.iterator().next();
      final Vec vec = data2DVector.apply(first);
      @SuppressWarnings({"unchecked"})
      final Set<T> cluster = new HashSet<T>();
      Vec centroid = VecTools.copy(vec);
      int changesCount;
      double maxDist = maxDist0 + (1 - maxDist0) * Math.max(0, (1 - Math.log(1000) / Math.log(dataSet.size())));
      do {
        changesCount = 0;
        final Vec nextCentroid = VecTools.copy(centroid);
        cluster.add(first);
        unclassified.remove(first);
        for (final T currentTerm : unclassified) {
          final Vec currentVec = data2DVector.apply(currentTerm);
          final double distance = 1 - VecTools.cosine(centroid, currentVec);
          count ++;
          if (distance < maxDist && !cluster.contains(currentTerm)) {
            changesCount++;
            VecTools.scale(nextCentroid, cluster.size());
            VecTools.append(nextCentroid, currentVec);
            VecTools.scale(nextCentroid, 1. / (cluster.size() + 1));
            cluster.add(currentTerm);
          }
          else if (distance >= maxDist && cluster.contains(currentTerm)) {
            changesCount++;
            VecTools.scale(nextCentroid, -cluster.size());
            VecTools.append(nextCentroid, currentVec);
            VecTools.scale(nextCentroid, -1. / (cluster.size() - 1));
            cluster.remove(currentTerm);
          }
        }
        centroid = nextCentroid;
        final VecIterator iter = centroid.nonZeroes();
        while (iter.advance()) {
          if (iter.value() < 0.01)
            iter.setValue(0);
        }
        maxDist *= 0.99;
      }
      while (changesCount > 0);
      clusters.add(cluster);
      unclassified.removeAll(cluster);
    }
//    LOG.debug("Multiplications " + count + " for " + dataSet.size() + " objects");
    return clusters;
  }
}
