package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
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
//    if (lastLambda1 == null)
    {
      lastLambda1 = new double[grid.size()];
      lastLambda2 = new double[grid.size()];

      for (int bfIndex = 0; bfIndex < grid.size(); bfIndex++) {
        final BFGrid.BinaryFeature bf = grid.bf(bfIndex);
        lastLambda1[bfIndex] = Math.sqrt(0.01 / IntStream.of(globalLoss.points()).mapToDouble(idx -> Math.abs(learn.data().get(idx, bf.findex) - bf.condition)).average().orElse(0));
        lastLambda2[bfIndex] = lastLambda1[bfIndex];
      }
    }

    final double[] mean = new double[depth + 1];
    final double[] lambda1 = new double[depth];
    final double[] lambda2 = new double[depth];
    final boolean[] mask = new boolean[depth];
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(depth);

    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    int[] points = ArrayTools.sequence(0, learn.length());
    mean[0] = mean(points, globalLoss);

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
        Pair<Double, Double> lambda = estimateLambda(finalPoints, bf, isRight[0], globalLoss, bds);
        lambda1[level] = lambda.first;
        lambda2[level] = lambda.second;
        {
          L2.Stat updatedStat = new L2.Stat(globalLoss.target());
          IntStream.of(finalPoints).forEach(idx -> {
            double probRight = probRight(x_i(bds, idx, bf.findex) - bf.condition, lambda1[level], lambda2[level]);
            updatedStat.append(idx, (isRight[0] ? probRight : 1 - probRight) * globalLoss.weight(idx));
          });
          System.out.println("Optimized lambda: " + lambda1[level] + " " + lambda2[level] + " score: " + (isRight[0] ? rightScore : leftScore) + " -> " + globalLoss.base().score(updatedStat));
        }
      });

      points = sampleSubset(bds, bestSplitBF, current.getPoints(), lambda1[level], lambda2[level], isRight[0]);
      mean[level + 1] = mean(points, globalLoss) - mean[level];
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

    return new ProbRegion(conditions, mask, mean, lambda1, lambda2);
  }

  @NotNull
  private int[] sampleSubset(BinarizedDataSet bds, BFGrid.BinaryFeature bestSplitBF,
                             int[] points, double lambda1, double lambda2, boolean isRight) {
    TIntArrayList newPoints = new TIntArrayList(points.length);

    for (int i = 0; i < points.length; i++) {
      final double diffX = x_i(bds, points[i], bestSplitBF.findex) - bestSplitBF.condition;
      final double prob = isRight ? probRight(diffX, lambda1, lambda2) : 1. - probRight(diffX, lambda1, lambda2);

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


  private double probRight(double diffX, double lambda1, double lambda2) {
//    return diffX <= 0 ? 0. : 1.;

//    final double exp = 1. / lambda * Math.exp(-lambda * diffX * diffX);
//    return diffX <= 0 ? exp : 1 - exp;

    return diffX > 0 ? 1. / (1. + Math.exp(-(lambda1 * lambda1) * diffX)) : 1. / (1. + Math.exp(-(lambda2 * lambda2) * diffX));
  }

  private double gradProbRight(double diffX, double lambda1, double lambda2) {
    final double v = probRight(diffX, lambda1, lambda2);
    return diffX > 0 ? v * (1 - v) * (- lambda1 * lambda1) : v * (1 - v) * (- lambda2 * lambda2);
  }

  private double x_i(BinarizedDataSet bds, int pointIdx, int findex) {
    final VecDataSet original = (VecDataSet)bds.original();
    return original.data().get(pointIdx, findex);
  }

  private double score(BinarizedDataSet bds, int[] points, WeightedLoss<? extends L2> loss,
                       boolean isRight, double lambda1, double lambda2, BFGrid.BinaryFeature bf) {
    Lock lock = new ReentrantLock();
    L2.Stat updatedStat = new L2.Stat(loss.target());
    IntStream.of(points).parallel().forEach(idx -> {
      double probRight = probRight(x_i(bds, idx, bf.findex) - bf.condition, lambda1, lambda2);
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


  double[] lastLambda1;
  double[] lastLambda2;

  @NotNull
  private Pair<Double, Double> estimateLambda(int[] points, BFGrid.BinaryFeature bf, boolean isRight, WeightedLoss<? extends L2> loss, BinarizedDataSet bds) {
    VecDataSet original = (VecDataSet)bds.original();
    final double maxDiff = IntStream.of(points).mapToDouble(idx -> Math.abs(original.data().get(idx, bf.findex) - bf.condition)).max().orElse(0);
    if (maxDiff == 0)
      throw new IllegalArgumentException();
    double current1 = lastLambda1[bf.bfIndex];// 1 / maxDiff;
    double current2 = lastLambda2[bf.bfIndex];// 1 / maxDiff;
    final double gradStep = 0.3;

    for (int sgdIter = 0; sgdIter < 200; sgdIter++) {
      if ((sgdIter) % 100 == 0) {
        System.out.println("\t l1 " + current1 + " l2 " + current2);
      }
      final double v = score(bds, points, loss, isRight, current1, current2, bf);
      final double vPrime1 = score(bds, points, loss, isRight, current1 + MathTools.EPSILON * Math.abs(current1), current2, bf);
      final double grad1 = (vPrime1 - v) / MathTools.EPSILON / Math.abs(current1);

      final double vPrime2 = score(bds, points, loss, isRight, current1, current2 + MathTools.EPSILON * Math.abs(current2), bf);
      final double grad2 = (vPrime2 - v) / MathTools.EPSILON / Math.abs(current2);

      current1 -= gradStep * grad1;
      if (current1 <= 0)
        current1 = MathTools.EPSILON;

      current2 -= gradStep * grad2;
      if (current2 <= 0)
        current2 = MathTools.EPSILON;
    }
    lastLambda1[bf.bfIndex] = current1;
    lastLambda2[bf.bfIndex] = current2;

    return new Pair<>(current1, current2);
  }

  public class ProbRegion extends FuncC1.Stub {
    private final BFGrid.BinaryFeature[] features;
    private final boolean[] mask;
    private final double[] mean;
    private final double[] lambda1;
    private final double[] lambda2;
    private final int depth;

    public ProbRegion(List<BFGrid.BinaryFeature> conditions, boolean[] mask, double[] mean,
                      double[] lambda1, double[] lambda2) {
      this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
      this.mask = mask;
      this.mean = mean;
      this.lambda1 = lambda1;
      this.lambda2 = lambda2;
      this.depth = features.length;
    }

    private double probLevel(int level, Vec x) {
      final double diffX = x.get(features[level].findex) - features[level].condition;
      return mask[level] ? probRight(diffX, lambda1[level], lambda2[level]) :
          (1. - probRight(diffX, lambda1[level], lambda2[level]));
    }

    @Override
    public double value(Vec x) {
      double result = mean[0];
      double prob = 1.;
      for (int level = 0; level < features.length; level++) {
        prob *= probLevel(level, x);
        result += mean[level + 1] * prob;
      }
      return result;
    }

    private double gradientByFeature(int level, int fIndex, Vec x) {
      if (features[level].findex != fIndex)
        return 0.;

      final double diffX = x.get(fIndex) - features[level].condition;
      final double v = gradProbRight(diffX, lambda1[level], lambda2[level]);
      return mask[level] ? v : -v;
    }

    @Override
    public Vec gradientTo(Vec x, Vec to) {
      final double[] beta = new double[depth];
      beta[depth - 1] = mean[depth];
      for (int level = depth - 2; level >= 0; level--) {
        beta[level] = mean[level + 1] + probLevel(level, x) * beta[level + 1];
      }

      for (int fIndex = 0; fIndex < x.dim(); fIndex++) {
        double grad = 0.;
        double prob = 1.;
        for (int level = 0; level < features.length; level++) {
          grad += prob * gradientByFeature(level, fIndex, x) * beta[level];
          prob *= probLevel(level, x);
        }
        to.set(fIndex, grad);
      }

      return to;
    }

    @Override
    public int dim() {
      return grid.rows();
    }
  }
}
