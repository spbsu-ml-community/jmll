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
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

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
  private int clustersCount = 1000;
  private Mx centroids;
  private Mx residuals; // смещение всех векторов относительно центроидов, для центроидов - нули.
  private Vec residualsNorm;
  private TIntArrayList vec2centr = new TIntArrayList(); // принадлженость центроиду (по номеру центроида)
  private TDoubleArrayList clustersSize = new TDoubleArrayList(); // размеры кластеров (по номеру центроида)

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
  private int zeroTransfers = 0;

  private void move(int i, int j, double weight) {
//    if ((-1 - 1000 * Math.log(rng.nextDouble())) > i)
//      return;
    if (i == j)
      return;
//    if (rng.nextDouble() * 0.9 * p(0) < p(i))
//      return;
//    weight *= Math.min(1, 1e-4 / p(i));
    final Vec v_i = symmetric(i, new ArrayVec(dim));
    final Vec v_j = symmetric(j, new ArrayVec(dim));
    final int cluster_i = vec2centr.get(i);
    final int cluster_j = vec2centr.get(j);
    { // removing i and j from their clusters
      removeFromCluster(i, cluster_i);
      removeFromCluster(j, cluster_j);
    }

    double product = multiply(v_i, v_j);
    final Vec grad = new ArrayVec(dim);
    final Vec weights = MxTools.multiply(centroids, v_i);
    double maxLogWeight = Math.max(product, max(weights));
    final double correctW;
    { // normalization for better arithmetic stability
      IntStream.range(0, weights.dim()).forEach(idx -> weights.set(idx, Math.exp(weights.get(idx) - maxLogWeight) * clustersSize.get(idx)));
      correctW = p(j) * Math.exp(product - maxLogWeight);
    }
    double denom = correctW + sum(weights);

    for (int k = 0; k < clustersCount; k++) {
      final double clusterGrad = -weights.get(k) / denom;
      if (Math.abs(clusterGrad) > 1e-5) {
        final Vec centroid_k = centroids.row(k);
        incscale(grad, centroid_k, clusterGrad);
        adaStep(vec2centr.size() + k, centroid_k, v_i, step(), weight * clusterGrad);
      }
    }
    incscale(grad, v_j, (1 - correctW / denom));
    adaStep(j, v_j, v_i, step(), weight * (1 - correctW / denom));
    adaStep(i, v_i, grad, step(), weight);

    final double score = (Math.log(p(j)) + product - maxLogWeight - Math.log(denom));
    this.score = this.score * G + score;

//    { // new score
//      final double new_correctW = p(j) * Math.exp(multiply(v_i, v_j));
//      double newDenom = new_correctW;
//      for (int k = 0; k < clustersCount; k++) {
//        final double w_k = clustersSize.get(k) * Math.exp(multiply(centroids.row(k), v_i));
//        newDenom += w_k;
//      }
//      final double newScore = weight * Math.log(new_correctW / newDenom);
//      if (score > newScore)
//        errors++;
//    }

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
    { // populating empty clusters
      final double minWeight = p(dict().size() - 1);
      Vec sampleWeight = new ArrayVec(residualsNorm.dim());
      assign(sampleWeight, residualsNorm);
      IntStream.range(0, residualsNorm.dim()).forEach(idx -> sampleWeight.set(idx, residualsNorm.get(idx) / p(idx)));
      IntStream.range(0, clustersCount).filter(idx -> clustersSize.get(idx) < minWeight).forEach(idx -> {
          symmetric(rng.nextSimple(sampleWeight), centroids.row(idx));
          populateCluster(idx);
      });
    }
    if ((++idx % 10000) == 0) {
      double residualsMean = 0;
      for (int u = 0; u < residuals.rows(); u++) {
        residualsMean += norm(residuals.row(u)) / residuals.rows();
      }
      System.out.print("\r" + idx + " score: " + CharSeqTools.ppDouble(score) + " total score: " + CharSeqTools.ppDouble(this.score / (1 - Math.pow(G, idx)) * (1 - G)) + " transfers: " + transfers + " residuals: " + residualsMean + " clusters: " + clustersCount);
      zeroTransfers += transfers == 0 ? 1 : 0;
      if (zeroTransfers > 100000) {
        zeroTransfers = 0;
        final int sample = rng.nextSimple(residualsNorm);
        clustersCount++;
        clustersSize.add(0);
        final VecBasedMx newCentroids = new VecBasedMx(clustersCount, dim);
        assign(newCentroids.vec.sub(0, centroids.dim()), centroids);
        symmetric(sample, newCentroids.vec.sub(centroids.dim(), dim));
        this.centroids = newCentroids;
        populateCluster(clustersCount - 1);
      }
      transfers = 0;
//      errors = 0;
    }
  }

  private void removeFromCluster(int index, int cluster) {
    vec2centr.set(index, -1);
    residualsNorm.set(index, 0);
    clustersSize.set(cluster, clustersSize.get(cluster) - p(index));
  }

  private void appendToCluster(int index, Vec v, int cluster) {
    if (cluster < 0)
      throw new IllegalArgumentException();
    vec2centr.set(index, cluster);
    final Vec centroid = centroids.row(cluster);
    clustersSize.set(cluster, clustersSize.get(cluster) + p(index));
    incscale(v, centroid, -1);
    assign(residuals.row(index), v);
    residualsNorm.set(index, sum2(v));
  }

  private void populateCluster(int cluster) {
    Vec to = new ArrayVec(dim);
    final Vec centroid = centroids.row(cluster);
    int population = 0;
    for (int i = 0; i < vec2centr.size(); i++) {
      if (vec2centr.get(i) < 0)
        continue;
      final Vec v = symmetric(i, to);
      if (distance(v, centroid) < Math.sqrt(residualsNorm.get(i))){
        population++;
        removeFromCluster(i, vec2centr.get(i));
        appendToCluster(i, v, cluster);
        to = new ArrayVec(dim);
      }
    }
    if (population == 0)
      System.out.println();
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
    if (Math.abs(scale) < 1e-8)
      return;

    while (accum.size() <= index) {
      accum.add(VecTools.fill(new ArrayVec(x.dim()), 1));
    }
    final Vec accum = this.accum.get(index);
    IntStream.range(0, x.dim()).forEach(i -> {
      final double increment = step * scale * grad.get(i) / Math.sqrt(accum.get(i));
      x.adjust(i, increment);
      accum.adjust(i, MathTools.sqr(scale * grad.get(i)));
    });
    VecTools.normalizeL2(x);
  }

  private void initialize() {
    // TODO: initialize centIds, vec2centr, centroidsSize
    final int vocSize = dict().size();
    final FastRandom rng = new FastRandom();
    centroids = new VecBasedMx(clustersCount, dim);
    residuals = new VecBasedMx(vocSize, dim);
    residualsNorm = new ArrayVec(vocSize);
    for (int k = 0; k < clustersCount; k++) {
      for (int j = 0; j < dim; j++) {
        centroids.set(k, j, 2 * (rng.nextDouble() - 0.5));
      }
      normalizeL2(centroids.row(k));
      clustersSize.add(0);
    }
    final int pointsPerCluster = vocSize / clustersCount;
    for (int i = 0; i < vocSize; i++) {
      final int cluster = Math.min(clustersCount - 1, i / pointsPerCluster);
      vec2centr.add(cluster);
      clustersSize.set(cluster, clustersSize.get(cluster) + p(i));
    }
  }
}
