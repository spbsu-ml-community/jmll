package com.spbsu.bernulli.dirichletMixture;

import com.spbsu.bernulli.Multinomial;
import com.spbsu.bernulli.caches.GammaCache;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

import static com.spbsu.bernulli.Utils.dist;

/**
 * Created by noxoomo on 06/04/15.
 */
public class DirichletEstimation {

  final int[] sums;
  final int[] counts;
  final FastRandom random;

  private final int[] clusters;
  private final int[] clustersSizes;

  private final LabelsManager labelsManager;

  private final int[] map;
  final double var;
  final double alpha = 0.5;
  final double beta = 0.5;
  final GammaCache gammaAlphaCache;
  final GammaCache gammaBetaCache;
  final GammaCache gammaAlphaBetaCache;

  final EnsembleEstimator estimator;

  //TODO: noxoomo: rewrite from prototype to good architecture with different distributions, etc
  public DirichletEstimation(double variance, int[] counts, int sums[], FastRandom random) {
    this.sums = sums;
    this.counts = counts;
    this.var = variance;
    this.random = new FastRandom(random.nextLong());
    //map for points
    this.clusters = new int[sums.length];
    this.clustersSizes = new int[sums.length];
    this.map = ArrayTools.sequence(0, sums.length);
    this.labelsManager = new LabelsManager(clusters.length);

    int maxCount = 0;
    for (int i = 0; i < clusters.length; ++i) {
      labelsManager.newLabel();
      clusters[i] = sums[i];
      clustersSizes[i] = counts[i];
      maxCount = Math.max(counts[i],maxCount);
    }

    this.gammaAlphaCache = new GammaCache(alpha, maxCount * sums.length);
    this.gammaBetaCache = alpha != beta ? new GammaCache(beta,maxCount * sums.length) : gammaAlphaCache;
    this.gammaAlphaBetaCache = new GammaCache(alpha + beta, maxCount * sums.length);
    estimator = new EnsembleEstimator(sums.length);
  }

  static int[] rep(int n, int times) {
    int[]  counts = new int[times];
    Arrays.fill(counts,n);
    return counts;
  }
  public DirichletEstimation(double variance, int count, int sums[], FastRandom random) {
    this(variance,rep(count,sums.length),sums,random);
  }

//  final public double[] estimation() {
//    double[] result = new double[sums.length];
//    for (int i = 0; i < sums.length; ++i) {
//      final int cluster = map[i];
//      result[i] = estimate(clusters[cluster], clustersSizes[cluster]);
//    }
//    return result;
//  }

  final public double[] estimate() {
    return estimator.get();
  }

  private double estimate(int m, int n) {
    return (m + alpha) / (n + alpha + beta);
  }


  private void sample() {
    for (int ind = 0; ind < sums.length; ++ind) {
      remove(ind);
//      sample(i);
      final int m = sums[ind];
      final int n = this.counts[ind];
      final int zeros = n - m;
      final int labelsCount = labelsManager.count();
      final double[] weights = new double[labelsCount + 1];
      for (int i = 0; i < weights.length - 1; ++i) {
        final int label = labelsManager.getLabel(i);
        final int heads = clusters[label];
        final int clSize = clustersSizes[label];
        final int tails = clSize - heads;
        final int w = clSize / n;
        final double tmp0 = gammaAlphaCache.logValue(m + heads) + gammaBetaCache.logValue(zeros + tails);
        final double tmp1 = gammaAlphaBetaCache.logValue(clSize) - gammaAlphaBetaCache.logValue(n + clSize);
        final double tmp2 = -gammaAlphaCache.logValue(heads) - gammaBetaCache.logValue(tails);
        weights[i] = w * FastMath.exp(tmp0 + tmp1 + tmp2);
      }

      {
        final double tmp0 = gammaAlphaCache.logValue(m) + gammaBetaCache.logValue(zeros);
        final double tmp1 = gammaAlphaBetaCache.logValue(0) - gammaAlphaBetaCache.logValue(n);
        final double tmp2 = -gammaAlphaCache.logValue(0) - gammaBetaCache.logValue(0);
        weights[weights.length - 1] = var * FastMath.exp(tmp0 + tmp1 + tmp2);
      }
      final int c = Multinomial.next(random, weights);
      if (c == labelsCount)
        createCluster(ind);
      else addToCluster(ind, labelsManager.getLabel(c));
    }
  }

  void run(int iterations) {
    run(iterations, -1);
  }

  final double eps = 1e-1;

  final public void burnIn() {
    final int window = 2;
    int iterations = 1000;
    double[] current = estimator.get();
    long it = 0;
    while (true) {
      estimator.clear();
      run(iterations, window);
      it += iterations;
      double[] next = estimator.get();
      double dist = dist(next, current);
      System.out.println("Estimation diff after  " + it + " iters is " + dist);
      if (dist < eps)
        return;
      current = next;
      iterations *= 2;
    }
  }

  final public void run(int iterations, int window) {
    if (window <= 0) {
      for (int i = 0; i < iterations; ++i) {
        sample();
      }
    } else {
      for (int i = 0; i < iterations; ++i) {
        sample();
        if (i % window == 0) {
          estimator.add(k -> {
            final int cluster = map[k];
            return estimate(clusters[cluster], clustersSizes[cluster]);
          });
        }
      }
    }
  }

  private void remove(int i) {
    final int label = map[i];
    clusters[label] -= sums[i];
    clustersSizes[label] -= counts[i];
    if (clustersSizes[label] == 0) {
      labelsManager.removeLabel(label);
    }
  }

  private void sample(int ind) {
    final int m = sums[ind];
    final int n = this.counts[ind];
    final int zeros = n - m;
    final int labelsCount = labelsManager.count();
    final double[] weights = new double[labelsCount + 1];
    for (int i = 0; i < weights.length - 1; ++i) {
      final int label = labelsManager.getLabel(i);
      final int heads = clusters[label];
      final int clSize = clustersSizes[label];
      final int tails = clSize - heads;
      final int w = clSize / n;
      final double tmp0 = gammaAlphaCache.logValue(m + heads) + gammaBetaCache.logValue(zeros + tails);
      final double tmp1 = gammaAlphaBetaCache.logValue(clSize) - gammaAlphaBetaCache.logValue(n + clSize);
      final double tmp2 = -gammaAlphaCache.logValue(heads) - gammaBetaCache.logValue(tails);
      weights[i] = w * FastMath.exp(tmp0 + tmp1 + tmp2);
    }

    {
      final double tmp0 = gammaAlphaCache.logValue(m) + gammaBetaCache.logValue(zeros);
      final double tmp1 = gammaAlphaBetaCache.logValue(0) - gammaAlphaBetaCache.logValue(n);
      final double tmp2 = -gammaAlphaCache.logValue(0) - gammaBetaCache.logValue(0);
      weights[weights.length - 1] = var * FastMath.exp(tmp0 + tmp1 + tmp2);
    }
    final int c = Multinomial.next(random, weights);
    if (c == labelsCount)
      createCluster(ind);
    else addToCluster(ind, labelsManager.getLabel(c));
  }

  private void addToCluster(int i, int label) {
    clusters[label] += sums[i];
    clustersSizes[label] += counts[i];
    map[i] = label;
  }

  private void createCluster(int i) {
    final int label = labelsManager.newLabel();
    clusters[label] = sums[i];
    clustersSizes[label] += counts[i];
    map[i] = label;
  }


  class LabelsManager {
    private int[] labels;
    private int[] labelsMap;
    private int last;

    public LabelsManager(int maxLabels) {
      labels = ArrayTools.sequence(0, maxLabels);
      labelsMap = ArrayTools.sequence(0, maxLabels);
      last = -1;
    }

    public int newLabel() {
      ++last;
      final int newLabel = labels[last];
      labelsMap[newLabel] = last;
      return newLabel;
    }

    public void removeLabel(int removedLabel) {
      final int lastLabel = labels[last];
      final int removedLabelIndex = labelsMap[removedLabel];
      labelsMap[lastLabel] = removedLabelIndex;
      labels[removedLabelIndex] = lastLabel;

      labelsMap[removedLabel] = last;
      labels[last] = removedLabel;
      --last;
    }

    public int count() {
      return last + 1;
    }

    public int getLabel(int i) {
      assert (i <= last);
      return labels[i];
    }
  }

}
