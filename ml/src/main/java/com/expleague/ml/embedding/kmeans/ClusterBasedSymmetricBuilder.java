package com.expleague.ml.embedding.kmeans;

import com.expleague.commons.math.MathTools;
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
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static com.expleague.commons.math.vectors.VecTools.*;

public class ClusterBasedSymmetricBuilder extends EmbeddingBuilderBase {
  private int dim = 50;
  private int clustersCount = 500;
  private Mx centroids;
  private Mx residuals; // смещение всех векторов относительно центроидов, для центроидов - нули.
  private Vec residualsNorm;
  private TIntArrayList vec2centr = new TIntArrayList(); // принадлженость центроиду (по номеру центроида)
  private TIntArrayList clustersSize = new TIntArrayList(); // размеры кластеров (по номеру центроида)

  private FastRandom rng = new FastRandom(100500);

  public ClusterBasedSymmetricBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  @SuppressWarnings("unused")
  public ClusterBasedSymmetricBuilder clustersNumber(int num) {
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
      result.put(dict().get(i), append(residuals.row(i), centroids.row(vec2centr.get(i))));
    }
    printClusters();
    return new EmbeddingImpl<>(result);
  }

  private Vec symmetric(int i, Vec to) {
    return VecTools.append(VecTools.assign(to, centroids.row(vec2centr.get(i))), residuals.row(i));
  }

  private double score = 0;
  private static double G = 0.99999;
  private long idx = 0;
  private int transfers = 0;
  private int errors = 0;
  private void move(int i, int j, double weight) {
    if ((-1 - 100 * Math.log(rng.nextDouble())) > i || (-1 - 100 * Math.log(rng.nextDouble())) > j)
      return;
    if (i == j)
      return;
    final Vec v_i = symmetric(i, new ArrayVec(dim));
    final Vec v_j = symmetric(j, new ArrayVec(dim));
    final int cluster_i = vec2centr.get(i);
    final int cluster_j = vec2centr.get(j);
    { // removing i and j from their clusters
      removeFromCluster(i, v_i, cluster_i);
      removeFromCluster(j, v_j, cluster_j);
    }

    final double correctW = Math.exp(multiply(v_i, v_j));
    final Vec grad = new ArrayVec(dim);
    final Vec weights = MxTools.multiply(centroids, v_i);
    VecTools.exp(weights);
    double denom = correctW;

    { // gradient for i
      for (int k = 0; k < clustersCount; k++) {
        if (clustersSize.get(k) == 0)
          continue;
        final double w_k = weights.get(k) * clustersSize.get(k);
        denom += w_k;
        if (w_k > 1e-8)
          incscale(grad, centroids.row(k), w_k);
      }
      scale(grad, -1./denom);
      if (Double.isFinite(correctW))
        incscale(grad, v_j, 1 - correctW / denom);
      adaStep(i, v_i, grad, step(), weight);
    }
    final double score = Double.isFinite(correctW) ? weight * Math.log(correctW / denom) : 0;
    this.score = this.score * G + score;

    for (int k = 0; k < clustersCount; k++) { // gradient for cluster k
      final Vec centroid_k = centroids.row(k);
      if (Double.isInfinite(weights.get(k)))
        continue;
      final double clusterGrad = -weights.get(k) * clustersSize.get(k) / denom;
      adaStep(vec2centr.size() + k, centroid_k, v_i, step(), weight * clusterGrad);
    }

    { // gradient for j
      if (!Double.isInfinite(correctW))
        adaStep(j, v_j, v_i, step(), weight * (1 - correctW / denom));
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
      final int newCluster_i = nearestCluster(v_i);
      if (newCluster_i != cluster_i)
        transfers++;
      appendToCluster(i, v_i, newCluster_i);
      final int newCluster_j = nearestCluster(v_j);
      if (newCluster_j != cluster_j)
        transfers++;
      appendToCluster(j, v_j, newCluster_j);
    }
    if ((++idx % 10000) == 0) {
      double sum = 0;
      int count = 0;
      for (int u = 0; u < centroids.rows(); u++) {
        for (int v = u + 1; v < centroids.rows(); v++, count++) {
          sum += VecTools.multiply(centroids.row(u), centroids.row(v));
        }
      }
      System.out.print("\r" + idx + " score: " + score + " total score: " + this.score / (1 - Math.pow(G, idx)) * (1 - G) + " transfers: " + transfers + " errors: " + errors + " cluster spread: " + (sum / count));
      transfers = 0;
      errors = 0;
    }
  }

  private void removeFromCluster(int index, Vec v, int cluster) {
    vec2centr.set(index, -1);
    residualsNorm.set(index, 0);
    final Vec centroid = centroids.row(cluster);
    scale(centroid, clustersSize.get(cluster));
    incscale(centroid, v, -1);
    final int newClusterSize = clustersSize.get(cluster) - 1;
    clustersSize.set(cluster, newClusterSize);
    if (clustersSize.get(cluster) > 0)
      scale(centroid, 1. / clustersSize.get(cluster));
    if (newClusterSize == 0) {
      boolean populated = false;
      while (!populated) {
        final int i = rng.nextSimple(residualsNorm);
        final int oldCluster = vec2centr.get(i);
        if (oldCluster < 0)
          continue;
        final Vec vec = symmetric(i, new ArrayVec(dim));
        removeFromCluster(i, vec, oldCluster);
        appendToCluster(i, vec, cluster);
        populated = true;
      }
    }
  }

  private void appendToCluster(int index, Vec v, int cluster) {
    if (cluster < 0)
      throw new IllegalArgumentException();
    vec2centr.set(index, cluster);
    final Vec centroid = centroids.row(cluster);
    scale(centroid, clustersSize.get(cluster));
    append(centroid, v);
    clustersSize.set(cluster, clustersSize.get(cluster) + 1);
    scale(centroid, 1. / clustersSize.get(cluster));
    incscale(v, centroid, -1);
    assign(residuals.row(index), v);
    residualsNorm.set(index, sum2(v));
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

  private void printClusters() {
    for (int i = 0; i < clustersCount; i++) {
      if (clustersSize.get(i) < 3)
        continue;
      for (int j = 0; j < vec2centr.size(); j++) {
        if (vec2centr.get(j) != i)
          continue;
        System.out.print(" " + dict().get(j));
      }
      System.out.println();
    }
  }

  private List<Vec> accum = new ArrayList<>();
  private void adaStep(int index, Vec x, Vec grad, double step, double scale) {
    if (scale < 1e-8)
      return;

    while (accum.size() <= index) {
      accum.add(VecTools.fill(new ArrayVec(x.dim()), 1));
    }
    final Vec accum = this.accum.get(index);
    double len = IntStream.range(0, x.dim()).mapToDouble(i -> {
      final double increment = step * scale * grad.get(i) / Math.sqrt(accum.get(i));
      x.adjust(i, increment);
      accum.adjust(i, MathTools.sqr(scale * grad.get(i)));
      return increment * increment;
    }).sum();
    if (Math.sqrt(len) > 10)
      System.out.println();
  }

  private void initialize() {
    // TODO: initialize centIds, vec2centr, centroidsSize
    final int voc_size = dict().size();
    final FastRandom rng = new FastRandom();
    centroids = new VecBasedMx(clustersCount, dim);
    residuals = new VecBasedMx(voc_size, dim);
    residualsNorm = new ArrayVec(voc_size);
    for (int k = 0; k < clustersCount; k++) {
      for (int j = 0; j < dim; j++) {
        centroids.set(k, j, 2 * (rng.nextDouble() - 0.5));
      }
      clustersSize.add(0);
    }
    for (int i = 0; i < voc_size; i++) {
      final int cluster = rng.nextInt(clustersCount);
      vec2centr.add(cluster);
      clustersSize.set(cluster, clustersSize.get(cluster) + 1);
    }
  }
}
