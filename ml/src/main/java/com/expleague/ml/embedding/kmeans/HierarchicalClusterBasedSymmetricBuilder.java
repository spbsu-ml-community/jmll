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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import com.expleague.commons.math.vectors.VecTools;

public class HierarchicalClusterBasedSymmetricBuilder extends EmbeddingBuilderBase {
  private int dim = 50;
  private int clustersCount = 300;
  private Mx residuals; // from the top level cluster
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
      result.put(dict().get(i), symmetric(i, new ArrayVec(dim)));
    }
    return new EmbeddingImpl<>(result);
  }

  private void initialize() {
    int voc_size = dict().size();
    clusters = new ClustersHierarcy(voc_size, 4);
    residuals = new VecBasedMx(voc_size, dim);
  }

  private Vec symmetric(int index, Vec to) {
    final Vec centroid = clusters.getClusters(index).row(0);
    return VecTools.append(VecTools.assign(to, centroid), residuals.row(index));
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
    final Mx centroids = clusters.getClusters(i);
    final TIntArrayList sizes = clusters.getClustersSizes(i);

    TIntArrayList oldPath_i = clusters.getPath(i, new TIntArrayList());
    TIntArrayList oldPath_j = clusters.getPath(j, new TIntArrayList());
    { // removing i and j from their clusters
      clusters.remove(i, v_i);
      clusters.remove(j, v_j);
    }

    final double correctW = Math.exp(VecTools.multiply(v_i, v_j));
    final Vec weights = MxTools.multiply(centroids, v_i);
    VecTools.exp(weights);
    double denom = correctW;

    { // gradient for i
      final Vec grad = new ArrayVec(dim);
      for (int k = 0; k < centroids.rows(); k++) {
        if (sizes.get(k) == 0)
          continue;
        final double w_k = weights.get(k) * sizes.get(k);
        denom += w_k;
        VecTools.incscale(grad, centroids.row(k), w_k);
      }
      VecTools.scale(grad, -1./denom);
      if (Double.isFinite(correctW))
        VecTools.incscale(grad, v_j, 1 - correctW / denom);
      adaStep(i, v_i, grad, step(), weight);
    }
    final double score = Double.isFinite(correctW) ? weight * Math.log(correctW / denom) : 0;
    this.score = this.score * G + score;

    final Vec old_top_centroid_i = centroids.row(0);
    final int top_centroid_i = oldPath_i.get(0);
    for (int k = 0; k < centroids.rows(); k++) { // gradient for cluster_i
      final double clusterGrad = Double.isFinite(weights.get(k)) ? -weights.get(k) * sizes.get(k) / denom : -1.;
      clusters.incscale(oldPath_i, k, v_i, step(), weight * clusterGrad);
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
      TIntArrayList newPath_i = clusters.append(i, v_i);
      final Vec new_top_cetroid_i = clusters.getClusters(i).row(0);
      for (int id = 0; id < dict().size(); id++) {
        if (clusters.getPath(id, new TIntArrayList()).get(0) == top_centroid_i) {
          Vec vec = new ArrayVec(dim);
          VecTools.append(VecTools.assign(vec, old_top_centroid_i), residuals.row(id));
          VecTools.incscale(vec, new_top_cetroid_i, -1);
          VecTools.assign(residuals.row(id), vec);
        }
      }
      //VecTools.incscale(v_i, clusters.getClusters(i).row(0), -1);
      //VecTools.assign(residuals.row(i), v_i);
      if (!newPath_i.equals(oldPath_i))
        transfers++;

      TIntArrayList newPath_j = clusters.append(j, v_j);
      VecTools.incscale(v_j, clusters.getClusters(j).row(0), -1);
      VecTools.assign(residuals.row(j), v_j);
      if (!newPath_j.equals(oldPath_j))
        transfers++;
    }
    if ((++idx % 10000) == 0) {
      System.out.print("\r" + idx + " score: " + score + " total score: " + this.score / (1 - Math.pow(G, idx)) * (1 - G) + " transfers: " + transfers + " errors: " + errors);
      transfers = 0;
      errors = 0;
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

  private class ClustersHierarcy {
    private List<List<ClustersLevel>> levels;
    private final int levelsNum;
    private final int levelSize;
    private final int voc_size;

    private final List<TIntArrayList> paths;

    public ClustersHierarcy(int voc_size, int levelSize) {
      levelsNum = (int) Math.floor(Math.log(clustersCount) / Math.log(levelSize)) + 1;
      this.levelSize = levelSize;
      this.voc_size = voc_size;
      levels = new ArrayList<>(levelsNum);
      for (int i = 0; i < levelsNum; i++) {
        int sz = (int) Math.pow(levelSize, i);
        List<ClustersLevel> tmp = new ArrayList<>(sz);
        for (int j = 0; j < sz; j++) {
          tmp.add(new ClustersLevel(i));
        }
        levels.add(tmp);
      }

      paths = new ArrayList<>(voc_size);
      final FastRandom rng = new FastRandom();
      for (int i = 0; i < voc_size; i++) {
        TIntArrayList path = new TIntArrayList(levelsNum);
        for (int j = 0; j < levelsNum; j++) {
          final int cluster = rng.nextInt(levelSize);
          path.add(cluster);
          int offset = getOffset(path, j);
          levels.get(j).get(offset).incClusterSize(cluster);
        }
        paths.add(path);
      }
      int y = 1;
    }

    public int getLevelsNum() {
      return levelsNum;
    }

    public TIntArrayList getPath(int index, TIntArrayList to) {
      to.addAll(paths.get(index));
      return to;
    }

    public void incscale(TIntArrayList path, int level, Vec grad, double step, double scale) {
      int adaId = voc_size;
      for (int i = 0; i < level; i++) {
        adaId += levels.get(i).size();
      }
      int cluster = path.get(level);
      int offset = getOffset(path, level);
      adaId += cluster;
      levels.get(level).get(offset).incscale(adaId, cluster, grad, step, scale);
    }

    public TIntArrayList append(int index, Vec v) {
      TIntArrayList path = new TIntArrayList(levelsNum);
      for (int i = 0; i < levelsNum; i++) {
        int offset = getOffset(path, i);
        int cluster = levels.get(i).get(offset).append(v);
        path.add(cluster);
      }
      paths.set(index, path);
      return path;
    }

    public TIntArrayList remove(int index, Vec v) {
      TIntArrayList path = paths.get(index);
      for (int i = 0; i < levelsNum; i++) {
        int offset = getOffset(path, i);
        levels.get(i).get(offset).remove(v, path.get(i));
      }
      paths.get(index).fill(-1);
      return path;
    }

    public Mx getClusters(int index) {
      Mx result = new VecBasedMx(levelsNum, dim);
      TIntArrayList path = paths.get(index);
      for (int i = 0; i < levelsNum; i++) {
        int offset = getOffset(path, i);
        Vec centroid = new ArrayVec(dim);
        levels.get(i).get(offset).getCluster(path.get(i), centroid);
        VecTools.assign(result.row(i), centroid);
      }
      return result;
    }

    public TIntArrayList getClustersSizes(int index) {
      TIntArrayList result = new TIntArrayList(levelsNum);
      TIntArrayList path = paths.get(index);
      for (int i = 0; i < levelsNum; i++) {
        int offset = getOffset(path, i);
        result.add(levels.get(i).get(offset).getClusterSize(path.get(i)));
      }
      return result;
    }

    private int getOffset(TIntArrayList path, int lastIndex) {
      int offset = 0;
      for (int j = 0; j < lastIndex; j++) {
        offset = offset * levelSize + path.get(j);
      }
      return offset;
    }

    private class ClustersLevel {
      private final int num;
      private Mx centroids;
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
      }

      protected void incClusterSize(int cluster) {
        clustersSize.set(cluster, clustersSize.get(cluster) + 1);
      }

      protected void incscale(int adaId, int cluster, Vec grad, double step, double scale) {
        Vec centroid = centroids.row(cluster);
        adaStep(adaId, centroid, grad, step, scale);
      }

      protected int append(Vec v) {
        final int newCluster = nearestCluster(v);
        appendToCluster(v, newCluster);
        return newCluster;
      }

      protected void remove(Vec v, int cluster) {
        removeFromCluster(v, cluster);
      }

      protected Vec getCluster(int cluster, Vec to) {
        return VecTools.assign(to, centroids.row(cluster));
      }

      protected int getClusterSize(int cluster) {
        return clustersSize.get(cluster);
      }


      private void removeFromCluster(Vec v, int cluster) {
        final Vec centroid = centroids.row(cluster);
        VecTools.scale(centroid, clustersSize.get(cluster));
        VecTools.incscale(centroid, v, -1);
        clustersSize.set(cluster, clustersSize.get(cluster) - 1);
        if (clustersSize.get(cluster) > 0)
          VecTools.scale(centroid, 1. / clustersSize.get(cluster));
      }

      private void appendToCluster(Vec v, int cluster) {
        final Vec centroid = centroids.row(cluster);
        VecTools.scale(centroid, clustersSize.get(cluster));
        VecTools.append(centroid, v);
        clustersSize.set(cluster, clustersSize.get(cluster) + 1);
        VecTools.scale(centroid, 1. / clustersSize.get(cluster));
      }

      private int nearestCluster(Vec v) {
        double minDistance = Double.POSITIVE_INFINITY;
        int result = -1;
        for (int i = 0; i < centroids.rows(); i++) {
          final Vec c_i =  centroids.row(i);
          final double distance = VecTools.distance(c_i, v);
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
