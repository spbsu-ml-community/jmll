package com.expleague.ml.embedding.kmeans;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.LongStream;

import static com.expleague.commons.math.vectors.VecTools.*;

public class HierarchicalClusterBasedSymmetricBuilder extends EmbeddingBuilderBase {
  private int dim = 5;
  private int clustersCount = 100;
  private Mx vectors;
  private ClustersHierarcy clusters;

  private FastRandom rng = new FastRandom(100500);

  public HierarchicalClusterBasedSymmetricBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  @SuppressWarnings("unused")
  public HierarchicalClusterBasedSymmetricBuilder clustersNumber(int num) {
    clustersCount = num;
    return this;
  }

  @Override
  protected Embedding<CharSeq> fit() {
    initialize();

    log.info("Started scanning corpus." + this.path);
    for (int t = 0; t < T(); t++) {
      System.out.println();
      log.info("\nEpoch " + t);
      try (final LongStream stream = positionsStream()) {
        stream.forEach(tuple -> {
          move(unpackA(tuple), unpackB(tuple), unpackWeight(tuple));
        });
      }
      catch (IOException e) {
        throw new RuntimeException("Error in source function occured\n" + e.getMessage());
      }
    }
    System.out.println();
    final Map<CharSeq, Vec> result = new HashMap<>();
    for (int i = 0; i < dict().size(); i++) {
      result.put(dict().get(i), vectors.row(i));
    }
    return new EmbeddingImpl<>(result);
  }

  private void initialize() {
    clusters = new ClustersHierarcy(4);
  }

  private double score = 0;
  private static double G = 0.99999;
  private long idx = 0;
  private int transfers = 0;
  private int errors = 0;
  private void move(int i, int j, double weight) {
    if ((-1 - 1000 * Math.log(rng.nextDouble())) > i || (-1 - 1000 * Math.log(rng.nextDouble())) > j)
      return;
    if (i == j)
      return;
    final Vec v_i = vectors.row(i);
    final Vec v_j = vectors.row(j);
    Mx centroids = clusters.getClustersAndSizes(i);
    final Mx sizes = centroids.sub(0, dim, centroids.length(), 1);
    centroids = centroids.sub(0, 0, centroids.length(), dim);

    TIntArrayList oldPath_i, oldPath_j;
    { // removing i and j from their clusters
      oldPath_i = clusters.remove(i, v_i);
      oldPath_j = clusters.remove(j, v_j);
    }

    final double correctW = Math.exp(multiply(v_i, v_j));
    final Vec weights = MxTools.multiply(centroids, v_i);
    VecTools.exp(weights);
    double denom = correctW;

    { // gradient for i
      final Vec grad = new ArrayVec(dim);
      for (int k = 0; k < centroids.length(); k++) {
        if (sizes.get(k, 0) == 0)
          continue;
        final double w_k = weights.get(k) * sizes.get(k, 0);
        denom += w_k;
        incscale(grad, centroids.row(k), w_k);
      }
      scale(grad, -1./denom);
      if (Double.isFinite(correctW))
        incscale(grad, v_j, 1 - correctW / denom);
      incscale(v_i, grad, step() * weight);
    }
    final double score = Double.isFinite(correctW) ? weight * Math.log(correctW / denom) : 0;
    this.score = this.score * G + score;

    for (int k = 0; k < centroids.length(); k++) { // gradient for cluster_i
      Vec centroid = centroids.row(k);
      final double clusterGrad = Double.isFinite(weights.get(k)) ? -weights.get(k) * sizes.get(k, 0) / denom : -1.;
      incscale(centroid, v_i, step() * weight * clusterGrad);
    }

    { // gradient for j
      if (!Double.isInfinite(correctW))
        incscale(v_j, v_i, step() * weight * (1 - correctW / denom));
    }

    //    { // new score
    //      final double new_correctW = Math.exp(multiply(v_i, v_j));
    //      double newDenom = new_correctW;
    //      for (int k = 0; k < clustersCount; k++) {
    //        final double w_k = clustersSize.get(k) * Math.exp(multiply(centroids.row(k), v_i));
    //        newDenom += w_k;
    //      }
    //      final double newScore = weight * Math.log(new_correctW / newDenom);
    //      if (score > newScore)
    //        errors++;
    //    }
    //
    { // updating clusters alignment
      TIntArrayList newPath_i = clusters.append(i, v_i);
      assign(vectors.row(i), v_i);
      if (!newPath_i.equals(oldPath_i))
        transfers++;
      TIntArrayList newPath_j = clusters.append(j, v_j);
      assign(vectors.row(j), v_j);
      if (!newPath_j.equals(oldPath_j))
        transfers++;
    }
    if ((++idx % 10000) == 0) {
      System.out.print("\r" + idx + " score: " + score + " total score: " + this.score / (1 - Math.pow(G, idx)) * (1 - G) + " transfers: " + transfers + " errors: " + errors);
      transfers = 0;
      errors = 0;
    }
  }

  private class ClustersHierarcy {
    private ClustersLevel levels;
    private final int levelsNum;
    private final int levelSize;

    public ClustersHierarcy(int levelSize) {
      levelsNum = (int) Math.floor(Math.log(clustersCount) / Math.log(levelSize));
      this.levelSize = levelSize;
      levels = new ClustersLevel(0);
    }

    public int getLevelsNum() {
      return levelsNum;
    }

    public TIntArrayList append(int index, Vec v) {
      TIntArrayList path = new TIntArrayList();
      return levels.append(index, v, path);
    }

    public TIntArrayList remove(int index, Vec v) {
      TIntArrayList path = new TIntArrayList();
      return levels.remove(index, v, path);
    }

    public Mx getClustersAndSizes(int index) {
      Mx result = new VecBasedMx(levelsNum, dim + 1); // последнее значение - размер кластера
      return levels.getClustersAndSizes(index, result);
    }

    private class ClustersLevel {
      private final List<ClustersLevel> childs;
      private final int num;
      private Mx centroids;
      private TIntArrayList vec2centr = new TIntArrayList(); // принадлженость центроиду (по номеру центроида)
      private TIntArrayList clustersSize = new TIntArrayList(); // размеры кластеров (по номеру центроида)

      public ClustersLevel(int lvlNum) {
        num = lvlNum;

        centroids = new VecBasedMx(levelSize, dim);
        for (int k = 0; k < levelSize; k++) {
          for (int j = 0; j < dim; j++) {
            centroids.set(k, j, 2 * (rng.nextDouble() - 0.5));
          }
          clustersSize.add(0);
        }

        if (num < levelsNum - 1) {
          childs = new ArrayList<>(levelSize);
          for (int i = 0; i < levelSize; i++) {
            childs.add(new ClustersLevel(num + 1));
          }
        } else {
          childs = null;
        }
      }

      protected TIntArrayList append(int index, Vec v, TIntArrayList path) {
        final int newCluster = nearestCluster(v);
        appendToCluster(index, v, newCluster);
        path.add(newCluster);
        if (num < levelsNum - 1) {
          childs.get(newCluster).append(index, v, path);
        }
        return path;
      }

      protected TIntArrayList remove(int index, Vec v, TIntArrayList path) {
        int cluster = vec2centr.get(index);
        removeFromCluster(index, v, cluster);
        path.add(cluster);
        if (num < levelsNum - 1) {
          childs.get(cluster).remove(index, v, path);
        }
        return path;
      }

      protected Mx getClustersAndSizes(int index, Mx to) {
        int cluster = vec2centr.get(index);
        for (int i = 0; i < dim; i++) {
          to.set(num, i, centroids.get(cluster, i));
        }
        to.set(num, dim, clustersSize.get(cluster));
        if (num < levelsNum - 1) {
          childs.get(cluster).getClustersAndSizes(index, to);
        }
        return to;
      }

      private void removeFromCluster(int index, Vec v, int cluster) {
        vec2centr.set(index, -1);
        final Vec centroid = centroids.row(cluster);
        scale(centroid, clustersSize.get(cluster));
        incscale(centroid, v, -1);
        clustersSize.set(cluster, clustersSize.get(cluster) - 1);
        if (clustersSize.get(cluster) > 0)
          scale(centroid, 1. / clustersSize.get(cluster));
      }

      private void appendToCluster(int index, Vec v, int cluster) {
        vec2centr.set(index, cluster);
        final Vec centroid = centroids.row(cluster);
        scale(centroid, clustersSize.get(cluster));
        VecTools.append(centroid, v);
        clustersSize.set(cluster, clustersSize.get(cluster) + 1);
        scale(centroid, 1. / clustersSize.get(cluster));
      }

      private int nearestCluster(Vec v) {
        double minDistance = Double.POSITIVE_INFINITY;
        int result = -1;
        for (int i = 0; i < centroids.rows(); i++) {
          final Vec c_i =  centroids.row(i);
          final double distance = distance(c_i, v);
          if (minDistance > distance) {
            minDistance = distance;
            result = i;
          }
        }
        return result;
      }

    }
  }
}
