package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.LinearRegion;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.List;

/**
 * User: nooxoomo
 */
public class GreedyTDLinearRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  private final int depth;
  private final double lambda;
  private final FastRandom random = new FastRandom();


  public GreedyTDLinearRegion(final BFGrid grid,
                              final int depth,
                              final double lambda) {
    this.grid = grid;
    this.depth = depth;
    this.lambda = lambda;
  }


  @Override
  public LinearRegion fit(final VecDataSet learn,
                          final Loss loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(depth);
    final boolean[] usedBF = new boolean[grid.size()];
    final List<Boolean> mask = new ArrayList<>();
    Vec bestSolution = new ArrayVec(0);

    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double currentScore = Double.POSITIVE_INFINITY;

    final boolean[] isRight = new boolean[grid.size()];
    final double[] scores = new double[grid.size()];
    final Vec[] solutions = new Vec[grid.size()];

    final int[] points = learnPoints(loss, learn);
    TDoubleArrayList sums = new TDoubleArrayList(depth);
    TDoubleArrayList weights = new TDoubleArrayList(depth);

    final BFOptimizationRegion current  = new BFOptimizationRegion(bds, loss, points);
    {
      AdditiveStatistics statistics = current.total();
      sums.add(sum(statistics));
      final double totalWeight = weight(statistics);
      weights.add(totalWeight);
    }


    for (int level = 0; level < depth; ++level) {
      current.visitAllSplits((bf, left, right) -> {
        if (usedBF[bf.bfIndex]) {
          scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          solutions[bf.bfIndex] = null;
        } else {
          final double leftScore;

          Vec leftBetas;

          final double leftWeight = weight(left);
          final double rightWeight = weight(right);
          final double minExcluded = Math.min(leftWeight, rightWeight);


          {
            if (minExcluded > 3) {
              final Vec regularizer = makeRegularizer(weights, leftWeight);
              Mx invCov = makeInvMatrix(weights, leftWeight, regularizer);
              Vec target = makeVector(sums, sum(left));
              Vec adjustTarget = adjustTarget(target, weights, leftWeight);
              leftBetas = MxTools.multiply(invCov, adjustTarget);
              leftScore = calcScore(invCov, target, leftBetas);
            } else {
              leftBetas = null;
              leftScore = Double.POSITIVE_INFINITY;
            }
          }

          Vec rightBetas;
          final double rightScore;
          {
            if (minExcluded > 3) {
              final Vec regularizer = makeRegularizer(weights, rightWeight);
              Mx invCov = makeInvMatrix(weights, rightWeight, regularizer);
              Vec target = makeVector(sums, sum(right));
              Vec adjustTarget = adjustTarget(target, weights, rightWeight);
              rightBetas = MxTools.multiply(invCov, adjustTarget);
              rightScore = calcScore(invCov, target, rightBetas);
            } else {
              rightBetas = null;
              rightScore = Double.POSITIVE_INFINITY;
            }
          }
          scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
          isRight[bf.bfIndex] = leftScore > rightScore;
          solutions[bf.bfIndex] = leftScore > rightScore ? rightBetas : leftBetas;
        }
      });

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0)
        break;


      if ((scores[bestSplit] >= currentScore))
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean bestSplitMask = isRight[bestSplitBF.bfIndex];


      conditions.add(bestSplitBF);
      usedBF[bestSplitBF.bfIndex] = true;
      mask.add(bestSplitMask);
      bestSolution = solutions[bestSplitBF.bfIndex];
      currentScore = scores[bestSplit];
      if (level < (depth - 1)) {
        current.split(bestSplitBF, bestSplitMask);

        final AdditiveStatistics total = current.total();
        sums.add(sum(total));
        final double weight = weight(total);
        weights.add(weight);
      }
    }

    final boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }

//
    final double bias = bestSolution.get(0);
    final double[] values = new double[bestSolution.dim() - 1];
    for (int i = 0; i < values.length; ++i) {
      values[i] = bestSolution.get(i + 1);
    }

    return new LinearRegion(conditions, masks, bias, values);
  }

  private Vec adjustTarget(Vec target, TDoubleArrayList weights, double weight) {
    final Vec adjusted = VecTools.copy(target);
    for (int i = 0; i < target.dim(); ++i) {
      final double w = i < weights.size() ? weights.get(i) : weight;
//      adjusted.set(i, target.get(i) * w / (w + 1));
      adjusted.set(i, target.get(i) * (w - 1) / w);
    }
    return adjusted;
  }


  private double calcScore(final Mx sigma, final Vec targetProj, final Vec betas) {
    final double targetBetasProd = VecTools.multiply(targetProj, betas);
    final Vec tmp = MxTools.multiply(sigma, targetProj);
    final double targetThroughInvSigmaDot = VecTools.multiply(targetProj, tmp);
//    final double rss = sum2 - 2 * targetThroughInvSigmaDot + targetBetasProd;
    return (0.5 * targetBetasProd - targetThroughInvSigmaDot);
//     return n * Math.log(rss / (n - targetProj.dim())) + betas.dim() * Math.log(n);
  }

  private Vec makeVector(TDoubleArrayList sums, double sum) {
    Vec result = new ArrayVec(sums.size() + 1);
    for (int i = 0; i < sums.size(); ++i) {
      result.set(i, sums.get(i));
    }
    result.set(sums.size(), sum);
    return result;
  }

  private Vec makeRegularizer(TDoubleArrayList weights, double weight) {
    final Vec reg = new ArrayVec(weights.size() + 1);
    VecTools.fill(reg, lambda);
//    reg.set(weights.size(), lambda);
    reg.set(0, 0);
    return reg;
  }

  private Mx makeMatrix(TDoubleArrayList weights, double weight) {
    final Mx cov = new VecBasedMx(weights.size() + 1, weights.size() + 1);
    final int n = weights.size() + 1;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        int idx = j < i ? i : j;
        cov.set(i, j, (idx < weights.size() ? weights.get(idx) : weight));
      }
    }
    return cov;
  }

  private Mx makeInvMatrix(TDoubleArrayList weights, double weight, Vec regularizer) {
    final Mx cov = new VecBasedMx(weights.size() + 1, weights.size() + 1);
    final int n = weights.size() + 1;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        int idx = j < i ? i : j;
        cov.set(i, j, (idx < weights.size() ? weights.get(idx) : weight));
      }
    }
    if (regularizer != null) {
      for (int i = 0; i < n; ++i) {
        cov.adjust(i, i, regularizer.get(i));
      }
    }
    return MxTools.inverseCholesky(cov);
  }

  class EmpericalBayesianLinearEstimator {
    private final Mx empericalCov;
    private final Vec targetProj;
    private final double sum2;
    private final double weight;

    private Mx posteriorCov;
    private Mx posteriorCovInv;
    private Vec mu;

    private Vec alphas;
    private double tau;

    private void updatePosteriors() {

      for (int i = 0; i < empericalCov.rows(); ++i) {
        for (int j = 0; j < empericalCov.columns(); ++j) {
          posteriorCov.set(i, j, tau * empericalCov.get(i, j));
          if (i == j) {
            posteriorCov.adjust(i, i, alphas.get(i));
          }
        }
      }
      posteriorCovInv = MxTools.inverseCholesky(posteriorCov);
      mu = MxTools.multiply(posteriorCovInv, targetProj);
      mu = VecTools.scale(mu, tau);
    }

    private void iterativeEstimate(int iterations) {

      for (int k = 0; k < iterations; ++k) {
        double N = weight;
        for (int i = 0; i < alphas.dim(); ++i) {
          final double gamma = 1.0 - alphas.get(i) * posteriorCovInv.get(i, i);
          N -= gamma;
          double val = gamma / MathTools.sqr(mu.get(i));
          if (val > 1000 || Double.isInfinite(val)) {
            val = 1000;
          }
          alphas.set(i, val);
        }
//        double sum = VecTools.sum(alphas);
//        VecTools.scale(alphas, 1e-10 / sum);

        double err = err();
        tau = N / err;

        updatePosteriors();
      }

//      updatePosteriors();
    }

    private void estimateAlpha() {
      for (int k = 0; k < 2; ++k) {
        double N = weight;
        final double w = empericalCov.get(0, empericalCov.rows() - 1);
        for (int i = 0; i < alphas.dim(); ++i) {
          if (i == (alphas.dim() - 1)) {
            final double q = tau * w - tau * tau * VecTools.sum(MxTools.multiply(posteriorCovInv, targetProj));
            final double s = tau * w - tau * tau * w * w * VecTools.sum(posteriorCovInv);
            if (q * q > s) {
              alphas.set(i, MathTools.sqr(s) / (MathTools.sqr(q) - s));
            } else {
              alphas.set(i, 1e10);
            }
          }
          final double gamma = 1.0 - alphas.get(i) * posteriorCovInv.get(i, i);
          N -= gamma;
        }
//          double val = gamma / MathTools.sqr(mu.get(i));
        double err = err();
        tau = N / err;
        updatePosteriors();
      }
    }

    EmpericalBayesianLinearEstimator(final double sum, double sum2, double weight,
                                     final Mx empericalCov,
                                     final Vec targetCov) {
      this.empericalCov = empericalCov;
      this.targetProj = targetCov;
      alphas = new ArrayVec(empericalCov.rows());
      VecTools.fill(alphas, 1e-3);
      final double var = (sum2 / weight) - MathTools.sqr(sum / weight);
      tau = 1.0 / var;
      this.sum2 = sum2;
      this.weight = weight;
      this.posteriorCov = new VecBasedMx(empericalCov);

      updatePosteriors();
      double err = err();
      tau = weight / err;
//      iterativeEstimate(5);
//      estimateAlpha();
    }

    public Vec mu() {
      return mu;
//      Mx cov = new VecBasedMx(empericalCov);
//      for (int i = 0; i < mu.dim(); ++i) {
////      final double preWeight = weights.get(i - 1);
////      final double w = empericalCov.get(i, i);
//        cov.adjust(i, i, alphas.get(i));
//      }
//      return MxTools.multiply(MxTools.inverseCholesky(cov), targetProj);
    }

    double score(Vec betas) {
      return err();
//      final double targetBetasProd = VecTools.multiply(targetProj, betas);
//      final Vec tmp = MxTools.multiply(MxTools.inverseCholesky(empericalCov), targetProj);
//      final double targetThroughInvSigmaDot = VecTools.multiply(targetProj, tmp);
//      final double rss = sum2 - 2 * targetThroughInvSigmaDot + targetBetasProd;
//    return 0.5 * targetBetasProd -  targetThroughInvSigmaDot;
//      return (0.5 * targetBetasProd - targetThroughInvSigmaDot);
    }

    double err() {
      double err = sum2;
      err -= 2 * tau * VecTools.multiply(targetProj, MxTools.multiply(posteriorCovInv, targetProj));
      err += VecTools.multiply(mu, MxTools.multiply(empericalCov, mu));
      assert (err > 0);
      return err;
    }
  }


  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }

  private double weight(final AdditiveStatistics stat) {
    if (stat instanceof L2.MSEStats) {
      return ((L2.MSEStats) stat).weight;
    } else if (stat instanceof WeightedLoss.Stat) {
      return weight(((WeightedLoss.Stat) stat).inside);
    } else {
      throw new RuntimeException("error");
    }
  }

  private double sum(final AdditiveStatistics stat) {
    if (stat instanceof L2.MSEStats) {
      return ((L2.MSEStats) stat).sum;
    } else if (stat instanceof WeightedLoss.Stat) {
      return sum(((WeightedLoss.Stat) stat).inside);
    } else {
      throw new RuntimeException("error");
    }
  }
}
