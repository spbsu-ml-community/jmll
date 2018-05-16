package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.stats.OrderByFeature;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.BFOptimizationSubset;
import gnu.trove.list.array.TIntArrayList;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 * Modified: 12.05.18 (Kruch.Dmitriy)
 * Based on GreedyTDProbRegion from nadya
 */
public class GreedyProbLinearRegion<Loss extends WeightedLoss<? extends L2>> extends VecOptimization.Stub<Loss> {
  private static final FastRandom rng = new FastRandom();
  protected final BFGrid grid;
  private final int depth;

  public GreedyProbLinearRegion(final BFGrid grid, int depth) {
    this.grid = grid;
    this.depth = depth;
  }

  @Override
  public ProbRegion fit(VecDataSet learn, final Loss globalLoss) {
    final double[] alpha = new double[depth + 1];
    final double[] lambda = new double[depth];
    final boolean[] mask = new boolean[depth];
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(depth);

    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    int[] points = ArrayTools.sequence(0, learn.length());
    alpha[0] = mean(points, globalLoss);

    BFOptimizationSubset current = new BFOptimizationSubset(bds, globalLoss, points);
    double currentScore = Double.POSITIVE_INFINITY;

    final double[] scores = new double[grid.size()];
    for (int l = 0; l < depth; l++) {
      final int level = l;

      final int[] finalPoints = points;
      current.visitAllSplits((bf, left, right) -> {
        final double leftScore = globalLoss.score((WeightedLoss.Stat) left);
        final double rightScore = globalLoss.score((WeightedLoss.Stat) right);
        scores[bf.bfIndex] = Math.min(leftScore, rightScore);
      });

      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;

      final boolean[] isRight = new boolean[1];
      BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      current.visitSplit(bestSplitBF, (bf, left, right) -> {
        final double leftScore = globalLoss.score((WeightedLoss.Stat) left);
        final double rightScore = globalLoss.score((WeightedLoss.Stat) right);
        isRight[0] = rightScore <= leftScore;
        lambda[level] = estimateLambda(finalPoints, bf, isRight[0], globalLoss, bds);
        {
          L2.Stat updatedStat = new L2.Stat(globalLoss.target());
          IntStream.of(finalPoints).forEach(idx -> {
            double probRight = probRight(x_i(bds, idx, bf.findex) - bf.condition, lambda[level]);
            updatedStat.append(idx, (isRight[0] ? probRight : 1 - probRight) * globalLoss.weight(idx));
          });
          System.out.println("Optimized lambda: " + lambda[level] + " score: " + (isRight[0] ? rightScore : leftScore) + " -> " + globalLoss.base().score(updatedStat));
        }
      });

      points = sampleSubset(bds, bestSplitBF, current.getPoints(), lambda[level], isRight[0]);
      alpha[level + 1] = mean(points, globalLoss) - alpha[level];
      mask[level] = isRight[0];
      conditions.add(bestSplitBF);
      currentScore = scores[bestSplit];

      current = new BFOptimizationSubset(bds, globalLoss, points);
    }
//    System.out.print("\nlambdas: ");
//    for (int i = 0; i < lambda.length; i++) {
//      System.out.printf("%.4f ", lambda[i]);
//    }
//    System.out.println();

    return new ProbRegion(conditions, mask, alpha, lambda);
  }

  @NotNull
  private int[] sampleSubset(BinarizedDataSet bds, BFGrid.BinaryFeature bestSplitBF,
                             int[] points, double lambda, boolean isRight) {
    TIntArrayList newPoints = new TIntArrayList();

    for (int i = 0; i < points.length; i++) {
      final double diffX = x_i(bds, points[i], bestSplitBF.findex) - bestSplitBF.condition;
      final double prob = isRight ? probRight(diffX, lambda) : 1. - probRight(diffX, lambda);

      if (rng.nextDouble() < prob) {
        newPoints.add(points[i]);
      }
    }

    return newPoints.toArray();
  }

  private double mean(int[] points, final WeightedLoss<? extends L2> curLoss) {
    final Vec target = curLoss.target();
    double sum = 0.;
    int count = 0;
    for (int i = 0; i < points.length; i++) {
      sum += target.get(points[i]) * curLoss.weight(points[i]);
      count += curLoss.weight(points[i]);
    }

    return count == 0 ? 0 : sum / count;
  }

  private static class Stat {
    final double leftScore;
    final double rightScore;
    final double leftLambda;
    final double rightLambda;

    private Stat(double leftScore, double rightScore, double leftLambda, double rightLambda) {
      this.leftScore = leftScore;
      this.rightScore = rightScore;
      this.leftLambda = leftLambda;
      this.rightLambda = rightLambda;
    }
  }



  private double probRight(double diffX, double lambda) {
//    return diffX <= 0 ? 0. : 1.;

//    final double exp = 1. / lambda * Math.exp(-lambda * diffX * diffX);
//    return diffX <= 0 ? exp : 1 - exp;

    return 1. / (1. + Math.exp(-lambda * diffX));
  }

  private double x_i(BinarizedDataSet bds, int pointIdx, int findex) {
    final VecDataSet original = (VecDataSet)bds.original();
    return original.data().get(pointIdx, findex);
  }

  private double score(BinarizedDataSet bds, int[] points, WeightedLoss<? extends L2> loss,
                       boolean isRight, double lambda, BFGrid.BinaryFeature bf) {
    Lock lock = new ReentrantLock();
    L2.Stat updatedStat = new L2.Stat(loss.target());
    IntStream.of(points).parallel().forEach(idx -> {
      double probRight = probRight(x_i(bds, idx, bf.findex) - bf.condition, lambda);
      //noinspection StatementWithEmptyBody
      while (!lock.tryLock());
      updatedStat.append(idx, (isRight ? probRight : 1 - probRight) * loss.weight(idx));
      lock.unlock();
    });
    return loss.base().score(updatedStat);
  }

  public double l2Reg(double weight, double sum) {
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }

  double clamp(double value) {
    return Math.max(Math.min(value, 100_000.0), 0.0);
  }

  @NotNull
  private double estimateLambda(int[] points, BFGrid.BinaryFeature bf, boolean isRight, WeightedLoss<? extends L2> loss, BinarizedDataSet bds) {
    VecDataSet original = (VecDataSet)bds.original();
    Vec x = original.data().row(bf.findex);
    final double gradStep = 0.3;
    final double minDiff = IntStream.of(points.length).mapToDouble(idx -> Math.abs(x.get(idx) - bf.condition)).min().orElse(0);
    if (minDiff == 0)
      throw new IllegalArgumentException();
    double current = 10 / minDiff;

    for (int sgdIter = 0; sgdIter < 1000; sgdIter++) {
      final double v = score(bds, points, loss, isRight, current, bf);
      final double vPrime = score(bds, points, loss, isRight, current + MathTools.EPSILON, bf);
      final double grad = (vPrime - v) / MathTools.EPSILON;
      current -= gradStep * grad;
      if (current <= 0)
        current = MathTools.EPSILON;
      if ((sgdIter + 1) % 100 == 0)
        System.out.println("\t" + current);
    }
    return current;
  }

  private class ProbRegion extends FuncC1.Stub {
    private final BFGrid.BinaryFeature[] features;
    private final boolean[] mask;
    private final double[] alpha;
    private final double[] lambda;

    public ProbRegion(List<BFGrid.BinaryFeature> conditions, boolean[] mask, double[] alpha, double[] lambda) {
      this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
      this.mask = mask;
      this.alpha = alpha;
      this.lambda = lambda;
    }

    @Override
    public double value(Vec x) {
      double result = alpha[0];
      double prob = 1.;
      for (int i = 0; i < features.length; i++) {
        final double diffX = x.get(features[i].findex) - features[i].condition;
        final double prob_i = mask[i] ? probRight(diffX, lambda[i]) : 1. - probRight(diffX, lambda[i]);
        prob *= prob_i;
        result += alpha[i + 1] * prob;
      }
      return result;
    }

    @Override
    public int dim() {
      return grid.rows();
    }
  }
}
