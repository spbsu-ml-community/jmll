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
import java.util.HashMap;
import java.util.Map;
import java.util.stream.LongStream;

import static com.expleague.commons.math.vectors.VecTools.*;

public class ClusterBasedSymmetricBuilder extends EmbeddingBuilderBase {
  private int dim = 5;
  private int clustersCount = 100;
  private Mx centroids;
  private Mx residuals; // смещение всех векторов относительно центроидов, для центроидов - нули.
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
    if ((-1 - 1000 * Math.log(rng.nextDouble())) > i || (-1 - 1000 * Math.log(rng.nextDouble())) > j)
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
        incscale(grad, centroids.row(k), w_k);
      }
      scale(grad, -1./denom);
      if (Double.isFinite(correctW))
        incscale(grad, v_j, 1 - correctW / denom);
      incscale(v_i, grad, step() * weight);
    }
    final double score = Double.isFinite(correctW) ? weight * Math.log(correctW / denom) : 0;
    this.score = this.score * G + score;

    for (int k = 0; k < clustersCount; k++) { // gradient for cluster k
      final Vec centroid_k = centroids.row(k);
      final double clusterGrad = Double.isFinite(weights.get(k)) ? -weights.get(k) / denom : -1.;
      incscale(centroid_k, v_i, step() * weight * clusterGrad);
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
      System.out.print("\r" + idx + " score: " + score + " total score: " + this.score / (1 - Math.pow(G, idx)) * (1 - G) + " transfers: " + transfers + " errors: " + errors);
      transfers = 0;
      errors = 0;
    }
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
    append(centroid, v);
    clustersSize.set(cluster, clustersSize.get(cluster) + 1);
    scale(centroid, 1. / clustersSize.get(cluster));
    incscale(v, centroid, -1);
    assign(residuals.row(index), v);
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

  private void initialize() {
    // TODO: initialize centIds, vec2centr, centroidsSize
    final int voc_size = dict().size();
    final FastRandom rng = new FastRandom();
    centroids = new VecBasedMx(clustersCount, dim);
    residuals = new VecBasedMx(voc_size, dim);
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
