package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
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
  public ProbRegion fit(VecDataSet learn, Loss loss) {
    final double[] alpha = new double[depth + 1];
    final double[] lambda = new double[depth];
    final boolean[] mask = new boolean[depth];
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(depth);

    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    int[] weights = extractWeights(loss);
    int[] points = ArrayTools.sequence(0, learn.length());
    alpha[0] = mean(points, loss);

    WeightedLoss<L2> curLoss = updateLoss(loss, learn, weights, alpha[0]);
    BFOptimizationSubset current = new BFOptimizationSubset(bds, curLoss, points);
    double currentScore = Double.POSITIVE_INFINITY;

    final double[] scores = new double[grid.size()];
    for (int l = 0; l < depth; l++) {
      final int level = l;
      System.out.println(level);
      final int[] curPoints = points;

      current.visitAllSplits((bf, left, right) -> {
        final Stat stat = estimateLambda(bds, curPoints, bf, loss);
        scores[bf.bfIndex] = Math.min(stat.leftScore, stat.rightScore);
      });

      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0)// || scores[bestSplit] >= currentScore)
        break;
      System.out.println(scores[bestSplit]);
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean[] isRight = new boolean[1];
      current.visitSplit(bestSplitBF, (bf, left, right) -> {
        final Stat stat = estimateLambda(bds, curPoints, bf, loss);
        isRight[0] = stat.rightScore < stat.leftScore;
        lambda[level] = stat.rightScore < stat.leftScore ? stat.rightLambda : stat.leftLambda;
      });

      points = sampleSubset(bds, bestSplitBF, current.getPoints(), weights, lambda[level], isRight[0]);

      alpha[level + 1] = mean(points, curLoss);
      mask[level] = isRight[0];
      conditions.add(bestSplitBF);
      currentScore = scores[bestSplit];

      curLoss = updateLoss(loss, learn, weights, alpha[level + 1]);
      current = new BFOptimizationSubset(bds, curLoss, points);
    }

    return new ProbRegion(conditions, mask, alpha, lambda);
  }

  @NotNull
  private WeightedLoss<L2> updateLoss(Loss loss, VecDataSet learn, int[] weights, double bias) {
    final Vec target = VecTools.assign(new ArrayVec(loss.target().dim()), loss.target());
    VecTools.adjust(target, -bias);

    final L2 curLoss = DataTools.newTarget(loss.base().getClass(), target, learn);
    return new WeightedLoss<>(curLoss, weights);
  }

  @NotNull
  private int[] sampleSubset(BinarizedDataSet bds, BFGrid.BinaryFeature bestSplitBF,
                             int[] points, int[] weights, double lambda, boolean isRight) {
    TIntArrayList newPoints = new TIntArrayList();

    for (int i = 0; i < points.length; i++) {
      final double diffX = border(bds, points[i], bestSplitBF.findex) - bestSplitBF.condition;
      final double prob = prob(diffX, lambda, isRight);

      int numPoints = weights[points[i]];
      for (int j = 0; j < numPoints; j++) {
        if (rng.nextDouble() < prob) {
          weights[points[i]]--;
        }
      }

      if (weights[points[i]] > MathTools.EPSILON) {
        newPoints.add(points[i]);
      }
    }

    return newPoints.toArray();
  }

  private double mean(int[] points, WeightedLoss<? extends L2> curLoss) {
    final Vec target = curLoss.target();
    double sum = 0.;
    int count = 0;
    for (int i = 0; i < points.length; i++) {
      sum += target.get(points[i]) * curLoss.weight(points[i]);
      count += curLoss.weight(points[i]);
    }

    return sum / (count + 1.);
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

  private double prob(double diffX, double lambda, boolean isRight) {
    final double prob = Math.exp(-lambda * diffX * diffX);
    return (diffX > 0) == isRight ? prob : 1. - prob;
  }

  private double border(BinarizedDataSet bds, int pointIdx, int findex) {
    int binNo = bds.bins(findex)[pointIdx];
    final double[] borders = grid.row(findex).borders;
    return binNo == borders.length ? borders[binNo - 1] : borders[binNo];
  }

  private double score(BinarizedDataSet bds, int[] points, Loss loss, boolean isRight,
                       double lambda, BFGrid.BinaryFeature bf) {
    final Vec target = loss.target();

    double wSumLeft = 0.;
    double sum2Left = 0.;
    double sumLeft = 0.;

    double wSumRight = 0.;
    double sum2Right = 0.;
    double sumRight = 0.;

    for (int i = 0; i < points.length; i++) {
      final double xi = border(bds, points[i], bf.findex);
      final double v = xi - bf.condition;
      final double yi = target.get(points[i]);

      final double prob = prob(v, lambda, isRight);
      if (v <= 0) {
        wSumLeft += prob;
        sumLeft += prob * yi;
        sum2Left += prob * yi * yi;
      } else {
        wSumRight += prob;
        sumRight += prob * yi;
        sum2Right += prob * yi * yi;
      }
    }

    final double leftDisp = sum2Left - sumLeft * sumLeft / (wSumLeft + 1.);
    final double rightDisp = sum2Right - sumRight * sumRight / (wSumRight + 1.);

    return leftDisp + rightDisp;
  }

  private int[] extractWeights(Loss loss) {
    int[] weights = new int[loss.dim()];
    for (int i = 0; i < loss.dim(); i++) {
      weights[i] = (int) loss.weight(i);
    }
    return weights;
  }

  double clamp(double value) {
    return Math.max(Math.min(value, 100.0), 0.0);
  }

  @NotNull
  private Stat estimateLambda(BinarizedDataSet bds, int[] points, BFGrid.BinaryFeature bf, Loss loss) {
    // find argmax by lambda
    double prevLeft = Double.NEGATIVE_INFINITY;
    double prevRight = Double.NEGATIVE_INFINITY;
    double lambdaLeft = 0.5;
    double lambdaRight = 0.5;
    final double EPS = 1e-6;
    final double sgdStep = 1e-1;

    for (int sgdIter = 0; sgdIter < 30; sgdIter++) {
      final double left1 = score(bds, points, loss, false, lambdaLeft - EPS, bf);
      final double left2 = score(bds, points, loss, false, lambdaLeft + EPS, bf);
      final double gradLeft = (left2 - left1) / (2 * EPS);

      final double right1 = score(bds, points, loss, true, lambdaRight - EPS, bf);
      final double right2 = score(bds, points, loss, true, lambdaRight + EPS, bf);
      final double gradRight = (right2 - right1) / (2 * EPS);
//      System.out.println("sgd ["+ sgdIter + "] " + left1 + " " + right1);

      lambdaLeft -= gradLeft * sgdStep;
      lambdaRight -= gradRight * sgdStep;
      lambdaLeft = clamp(lambdaLeft);
      lambdaRight = clamp(lambdaRight);

      if (Math.abs(lambdaLeft - prevLeft) <= sgdStep / 1000
          || Math.abs(lambdaRight - prevRight) <= sgdStep / 1000) {
        break;
      }

      prevLeft = lambdaLeft;
      prevRight = lambdaRight;
    }

    double leftScore = score(bds, points, loss, false, lambdaLeft, bf);
    double rightScore = score(bds, points, loss, true, lambdaRight, bf);
    if (leftScore == 0.) {
      leftScore = Double.POSITIVE_INFINITY;
    }
    if (rightScore == 0.) {
      rightScore = Double.POSITIVE_INFINITY;
    }

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
      for (int i = 0; i < features.length; i++) {
        final double diffX = x.get(features[i].findex) - features[i].condition;
        final double prob = prob(diffX, lambda[i], mask[i]);
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
