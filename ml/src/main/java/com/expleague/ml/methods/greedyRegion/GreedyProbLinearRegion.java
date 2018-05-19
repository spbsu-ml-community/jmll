package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
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
import java.util.concurrent.atomic.DoubleAccumulator;
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
    lastLambda1 = new Vec[grid.size()];

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
        Vec lambda = estimateLambda(finalPoints, bf, isRight[0], globalLoss, bds);
        lambda1[level] = lambda.get(0);
        lambda2[level] = lambda.get(1);
        {
          L2.Stat updatedStat = new L2.Stat(globalLoss.target());
          IntStream.of(finalPoints).forEach(idx -> {
            double probRight = probRight(x_i(bds, idx, bf.findex) - bf.condition, lambda1[level], lambda2[level]);
            updatedStat.append(idx, (isRight[0] ? probRight : 1 - probRight) * globalLoss.weight(idx));
          });
//          System.out.println("Optimized lambda: " + lambda1[level] + " " + lambda2[level] + " score: " + (isRight[0] ? rightScore : leftScore) + " -> " + globalLoss.base().score(updatedStat));
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


  private static double probRight(double diffX, double lambda1, double lambda2) {
//    return diffX <= 0 ? 0. : 1.;

//    final double exp = 1. / lambda * Math.exp(-lambda * diffX * diffX);
//    return diffX <= 0 ? exp : 1 - exp;

    return diffX > 0 ? 1. / (1. + Math.exp(-lambda1 * lambda1 * diffX)) : 1. / (1. + Math.exp(-lambda2 * lambda2 * diffX));
  }

  private double gradProbRight(double diffX, double lambda1, double lambda2) {
    final double v = probRight(diffX, lambda1, lambda2);
    return diffX > 0 ? v * (1 - v) * (- lambda1 * lambda1) : v * (1 - v) * (- lambda2 * lambda2);
  }

  private static double x_i(BinarizedDataSet bds, int pointIdx, int findex) {
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


  Vec[] lastLambda1;

  @NotNull
  private Vec estimateLambda(int[] points, BFGrid.BinaryFeature bf, boolean isRight, WeightedLoss<? extends L2> loss, BinarizedDataSet bds) {
//    final ScoreFromLambda sfl = new ScoreFromLambda(bds, points, loss, isRight, bf);
//
//    Vec cursor = lastLambda1[bf.bfIndex] != null ? lastLambda1[bf.bfIndex] : (lastLambda1[bf.bfIndex] = new ArrayVec(1, 1));
//    Vec L = new ArrayVec(0.2, 0.2);
//    Vec grad = new ArrayVec(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
//    Vec step = new ArrayVec(2);
//    int iter = 0;
//    while (iter < 100000) {
//      sfl.gradientTo(cursor, grad);
//      if (VecTools.l2(grad) < 1e-12)
//        break;
//
//      VecTools.assign(step, grad);
//      VecTools.scale(step, L);
//      VecTools.incscale(cursor, step, 0.01);
//      for (int i = 0; i < L.dim(); i++) {
//        L.set(i, Math.min(L.getCached(i) / 0.999, 1 / grad.getCached(i)));
//      }
//      iter++;
//      if (iter % 1000 == 0)
//        System.out.println(cursor + " score: " + sfl.value(cursor));
//    }

    //        LOG.message("GDM iterations = " + iter + "\n\n");
    return new ArrayVec(1., 1.);
//
//    GradientDescent descent = new GradientDescent(new ArrayVec(0, 0), 0.00001);
//    return descent.optimize(sfl);
//    final Vec l = new ArrayVec(1, 1);
    //    final double rightBorderL1 = sfl.findRight(l, 0);
//    final double lambda1 = MathTools.bisection(new AnalyticFunc.Stub() {
//      @Override
//      public double value(double x) {
//        l.set(0, x);
//        return sfl.gradient(l).getCached(0);
//      }
//    }, -rightBorderL1, rightBorderL1);
//    l.set(0, lambda1);
//    final double rightBorderL2 = sfl.findRight(l, 1);
//    final double lambda2 = MathTools.bisection(new AnalyticFunc.Stub() {
//      @Override
//      public double value(double x) {
//        l.set(1, x);
//        return sfl.gradient(l).getCached(1);
//      }
//    }, -rightBorderL2, rightBorderL2);
//    l.set(1, lambda2);
//
//    return l;
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

      IntStream.range(0, x.dim()).parallel().forEach(fIndex -> {
        double grad = 0.;
        double prob = 1.;
        for (int level = 0; level < features.length; level++) {
          grad += prob * gradientByFeature(level, fIndex, x) * beta[level];
          prob *= probLevel(level, x);
        }
        to.set(fIndex, grad);
      });

      return to;
    }

    @Override
    public int dim() {
      return grid.rows();
    }
  }

  public static class ScoreFromLambda extends FuncC1.Stub {
    private final BinarizedDataSet bds;
    private final int[] points;
    private final WeightedLoss<? extends L2> loss;
    private final boolean isRight;
    private final BFGrid.BinaryFeature bf;

    public ScoreFromLambda(BinarizedDataSet bds, int[] points, WeightedLoss<? extends L2> loss, boolean isRight, BFGrid.BinaryFeature bf) {
      this.bds = bds;
      this.points = points;
      this.loss = loss;
      this.isRight = isRight;
      this.bf = bf;
    }

    @Override
    public double value(Vec x) {
      final double lambda1 = x.get(0);
      final double lambda2 = x.get(1);
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

    @Override
    public Vec gradientTo(Vec x, Vec to) {
      final DoubleAccumulator sumYAcc = new DoubleAccumulator((left, right) -> left + right, 0);
      final DoubleAccumulator sumWAcc = new DoubleAccumulator((left, right) -> left + right, 0);
      final double lambda1 = x.get(0);
      final double lambda2 = x.get(1);

      IntStream.of(points).parallel().forEach(idx -> {
        final double diffX = x_i(bds, idx, bf.findex) - bf.condition;
        final double probRight = probRight(diffX, lambda1, lambda2);
        final double w_i = probRight * loss.weight(idx);
        final double y_i = loss.target().get(idx);
        sumWAcc.accumulate(w_i);
        sumYAcc.accumulate(w_i * y_i);
      });

      final double sumW = sumWAcc.doubleValue();
      final double sumY = sumYAcc.doubleValue();

      final DoubleAccumulator sumLambda1 = new DoubleAccumulator((left, right) -> left + right, 0);
      final DoubleAccumulator sumLambda2 = new DoubleAccumulator((left, right) -> left + right, 0);
      IntStream.of(points).parallel().forEach(idx -> {
        final double diffX = x_i(bds, idx, bf.findex) - bf.condition;
        final double probRight = probRight(diffX, lambda1, lambda2);
        final double y_i = loss.target().get(idx);
        double dTdw_i = - (sumY * sumY - 2 * y_i * sumW * sumY);
        if (diffX >= 0)
          sumLambda1.accumulate(diffX * probRight * (1 - probRight) * dTdw_i);
        else
          sumLambda2.accumulate(diffX * probRight * (1 - probRight) * dTdw_i);
      });
      to.set(0, 2 * lambda1 * sumLambda1.doubleValue() / sumW / sumW);
      to.set(1, 2 * lambda2 * sumLambda2.doubleValue() / sumW / sumW);
      return to;
    }

    @Override
    public int dim() {
      return 2;
    }

    @Override
    public Vec L(Vec at) {
      return new ArrayVec(2, 2);
    }

    public double findRight(Vec l, int i) {
      l.set(i, 0);
      Vec to = new ArrayVec(dim());
      gradientTo(l, to);
      l.set(i, 1);
      double v_0 = to.get(i);
      while (v_0 * gradientTo(l, to).get(i) > 0) {
        l.set(i, 2 * l.get(i));
      }
      return l.get(i);
    }
  }
}
