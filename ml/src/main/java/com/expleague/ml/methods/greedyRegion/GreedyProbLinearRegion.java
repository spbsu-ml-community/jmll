package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
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

//        final Stat stat = estimateLambda(bds, finalPoints, bf, globalLoss);
//        scores[bf.bfIndex] = Math.min(stat.leftScore, stat.rightScore);
      });

      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;

      final boolean[] isRight = new boolean[1];
      BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      current.visitSplit(bestSplitBF, (bf, left, right) -> {
        final double leftScore = globalLoss.score((WeightedLoss.Stat) left);
        final double rightScore = globalLoss.score((WeightedLoss.Stat) right);
        isRight[0] = rightScore < leftScore;
        lambda[level] = 10.;

//        final Stat stat = estimateLambda(bds, finalPoints, bf, globalLoss);
//        isRight[0] = stat.rightScore < stat.leftScore;
//        lambda[level] = stat.rightScore < stat.leftScore ? stat.rightLambda : stat.leftLambda;
//        scores[bestSplit] = stat.rightScore < stat.leftScore ? stat.rightScore : stat.leftScore;
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
    return diffX <= 0 ? 0. : 1.;

//    final double exp = 1. / lambda * Math.exp(-lambda * diffX * diffX);
//    return diffX <= 0 ? exp : 1 - exp;

//    return 1. / (1. + Math.exp(-lambda * diffX));
  }

  private double x_i(BinarizedDataSet bds, int pointIdx, int findex) {
    final VecDataSet original = (VecDataSet)bds.original();
    return original.data().get(pointIdx, findex);
  }

  private double condition(BFGrid.BinaryFeature bf, boolean isRight) {
    return bf.condition;
//    final int bfStart = bf.row().bfStart;
//    final int bfEnd = bf.row().bfEnd;
//    final double[] borders = bf.row().borders;
//    if (borders.length > 1) {
//      if (isRight && (bf.bfIndex + 1) < bfEnd) {
//        return (borders[bf.bfIndex - bfStart] + borders[bf.bfIndex + 1 - bfStart]) / 2.;
//      } else if (!isRight && (bf.bfIndex - 1) >= bfStart) {
//        return (borders[bf.bfIndex - bfStart] + borders[bf.bfIndex - 1 - bfStart]) / 2.;
//      } else {
//        return bf.condition;
//      }
//    } else {
//      return bf.condition + 0.5;
//    }
  }

  private double score(BinarizedDataSet bds, int[] points, WeightedLoss<? extends L2> loss,
                       boolean isRight, double lambda, BFGrid.BinaryFeature bf) {
    final Vec target = loss.target();
    double wSum = 0.;
    double sum = 0.;

    for (int i = 0; i < points.length; i++) {
      final double weight = loss.weight(points[i]);
      final double diffX = x_i(bds, points[i], bf.findex) - condition(bf, isRight);
      final double yi = target.get(points[i]);

      final double prob = isRight ? probRight(diffX, lambda) : 1. - probRight(diffX, lambda);
      final double w = weight * prob;
      wSum += w;
      sum += w * yi;
    }

//    return l2Reg(wSum, sum);
    return wSum > 2 ? -sum * sum / wSum /* * (1 + 2 * Math.log(wSum + 1.)) */ : 0;
  }

  public double l2Reg(double weight, double sum) {
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }

  double clamp(double value) {
    return Math.max(Math.min(value, 100_000.0), 0.0);
  }

  @NotNull
  private synchronized Stat estimateLambda(BinarizedDataSet bds, int[] points, BFGrid.BinaryFeature bf, WeightedLoss<? extends L2> loss) {
    // find argmax by lambda
    double prevLeft = Double.NEGATIVE_INFINITY;
    double prevRight = Double.NEGATIVE_INFINITY;
    double lambdaLeft = 10;
    double lambdaRight = 10;
    final double EPS = 1e-6;
    double sgdStepLeft = 1e-2;
    double sgdStepRight = 1e-2;
    double lipshitzLeft = Double.POSITIVE_INFINITY;
    double lipshitzRight = Double.POSITIVE_INFINITY;

    for (int sgdIter = 0; sgdIter < 1000; sgdIter++) {
      final double left1 = score(bds, points, loss, false, lambdaLeft - EPS, bf);
      final double left2 = score(bds, points, loss, false, lambdaLeft + EPS, bf);
      final double gradLeft = (left2 - left1) / (2 * EPS);

      final double right1 = score(bds, points, loss, true, lambdaRight - EPS, bf);
      final double right2 = score(bds, points, loss, true, lambdaRight + EPS, bf);
      final double gradRight = (right2 - right1) / (2 * EPS);
//      System.out.println("sgd ["+ sgdIter + "] " + left1 + " " + right1 + " l " + lambdaLeft + " r " + lambdaRight);

      lambdaLeft -= gradLeft * sgdStepLeft;
      lambdaRight -= gradRight * sgdStepRight;

      lambdaLeft = clamp(lambdaLeft);
      lambdaRight = clamp(lambdaRight);

      if (sgdIter % 30 != 0) {
        lipshitzLeft = Math.min(Math.abs(gradLeft), lipshitzLeft);
        lipshitzRight = Math.min(Math.abs(gradRight), lipshitzRight);
      } else if (sgdIter != 0) {
        sgdStepLeft = 1e-2 / lipshitzLeft;
        sgdStepRight = 1e-2 / lipshitzRight;
        lipshitzLeft = Double.POSITIVE_INFINITY;
        lipshitzRight = Double.NEGATIVE_INFINITY;
      }

      if (Math.abs(lambdaLeft - prevLeft) <= MathTools.EPSILON
          || Math.abs(lambdaRight - prevRight) <= MathTools.EPSILON) {
        break;
      }

      prevLeft = lambdaLeft;
      prevRight = lambdaRight;
    }

    double leftScore = score(bds, points, loss, false, lambdaLeft, bf);
    double rightScore = score(bds, points, loss, true, lambdaRight, bf);
//    if (leftScore == 0.) {
//      leftScore = Double.POSITIVE_INFINITY;
//    }
//    if (rightScore == 0.) {
//      rightScore = Double.POSITIVE_INFINITY;
//    }

    return new Stat(leftScore, rightScore, lambdaLeft, lambdaRight);
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
        final double diffX = x.get(features[i].findex) - condition(features[i], mask[i]);
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
