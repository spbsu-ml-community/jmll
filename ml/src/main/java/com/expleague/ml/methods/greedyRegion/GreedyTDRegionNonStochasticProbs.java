package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.BFOptimizationSubset;
import com.expleague.ml.models.Region;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegionNonStochasticProbs<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  int maxFailed = 1;
  private double alpha = 0.08;
  private double beta = 1.1;
  private final FastRandom rand = new FastRandom();

  public GreedyTDRegionNonStochasticProbs(final BFGrid grid) {
    this.grid = grid;
  }

  public GreedyTDRegionNonStochasticProbs(final BFGrid grid, final double alpha, final double beta) {
    this(grid);
    this.alpha = alpha;
    this.beta = beta;
  }

  Pair<BFGrid.Feature[], boolean[]> initFit(final VecDataSet learn, final Loss loss) {
    final BFOptimizationSubset current;
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    current = new BFOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.length()));
    final double[] bestRowScores = new double[grid.rows()];
    for (int i = 0; i < bestRowScores.length; ++i) {
      bestRowScores[i] = Double.POSITIVE_INFINITY;
    }
    final BFGrid.Feature[] bestRowFeatures = new BFGrid.Feature[grid.rows()];
    final boolean[] masks = new boolean[grid.rows()];

    current.visitAllSplits((bf, left, right) -> {
      final double leftScore = logScore(left);
      final double rightScore = logScore(right);
      final double bestScore = leftScore > rightScore ? rightScore : leftScore;

      final int findex = bf.findex();
      if (bestScore < bestRowScores[findex]) {
        bestRowScores[findex] = bestScore;
        masks[findex] = leftScore > rightScore;
        bestRowFeatures[findex] = bf;
      }
    });


    final boolean[] resultMasks = new boolean[maxFailed];
    final BFGrid.Feature[] resultFeatures = new BFGrid.Feature[maxFailed];

    for (int i = 0; i < maxFailed; ) {
      final boolean[] used = new boolean[bestRowScores.length];
      final int index = rand.nextInt(bestRowScores.length);
      if (bestRowScores[index] < Double.POSITIVE_INFINITY && !used[index]) {
        used[index] = true;
        final BFGrid.Feature feature = bestRowFeatures[index];
        final boolean mask = masks[index];
        resultFeatures[i] = feature;
        resultMasks[i] = mask;
        ++i;
      }
    }
    return new Pair<>(resultFeatures, resultMasks);
  }

  @Override
  public Region fit(final VecDataSet learn, final Loss loss) {
//    maxFailed = rand.nextInt(4);
    final List<BFGrid.Feature> conditions = new ArrayList<>(32);
    final boolean[] usedBF = new boolean[grid.size()];
    final List<Boolean> mask = new ArrayList<>();
    final TDoubleArrayList conditionSum = new TDoubleArrayList(32);
    final TDoubleArrayList conditionTotal = new TDoubleArrayList(32);
    final Pair<BFGrid.Feature[], boolean[]> init = initFit(learn, loss);
    for (int i = 0; i < init.first.length; ++i) {
      conditions.add(init.first[i]);
      usedBF[init.first[i].index()] = true;
      mask.add(init.second[i]);
    }
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double currentScore = Double.POSITIVE_INFINITY;
    final BFWeakConditionsStochasticOptimizationRegion current =
            new BFWeakConditionsStochasticOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.first, init.second, maxFailed);
    current.alpha = alpha;
    current.beta = beta;
    final BFWeakConditionsOptimizationRegion determenisticCurrent =
            new BFWeakConditionsOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.first, init.second, maxFailed);

    final double[] scores = new double[grid.size()];
    final boolean[] isRight = new boolean[grid.size()];
    final double[] weights = new double[grid.size() * 2];


    while (true) {
      final double total = weight(determenisticCurrent.total());
      final double[] csum = new double[conditionSum.size() + 1];
      final double[] ctotal = new double[conditionSum.size() + 1];
      for (int i = 0; i < csum.length - 1; ++i) {
        csum[i] = conditionSum.get(i);
        ctotal[i] = conditionTotal.get(i);
      }

      determenisticCurrent.visitAllSplits((bf, left, right) -> {
        final AdditiveStatistics leftIn = (AdditiveStatistics) loss.statsFactory().create();
        final AdditiveStatistics rightIn = (AdditiveStatistics) loss.statsFactory().create();
        leftIn.append(left);
        leftIn.append(current.nonCriticalTotal);
        rightIn.append(right);
        rightIn.append(current.nonCriticalTotal);
        final double leftWeight = weight(leftIn);
        final double rightWeight = weight(rightIn);
        weights[bf.index() * 2] = leftWeight;
        weights[bf.index() * 2 + 1] = rightWeight;
      });


      current.visitAllSplits((bf, left, right) -> {
        if (usedBF[bf.index()]) {
          scores[bf.index()] = Double.POSITIVE_INFINITY;
        } else {
          double leftScore;
          {
            csum[csum.length - 1] = weights[bf.index() * 2];
            ctotal[csum.length - 1] = total;
            final double prob = estimate(csum, ctotal);
            final AdditiveStatistics out = (AdditiveStatistics) loss.statsFactory().create();
            final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
            out.append(right);
            out.append(current.excluded);
            in.append(current.nonCriticalTotal);
            in.append(left);
            leftScore = (1 - prob) * outScore(out) + prob * inScore(in);// + 2 * ((1-prob) * Math.log(1-prob) + prob * Math.log(prob));
//              leftScore = score(out) +  score(in);// + 2 * ((1-prob) * Math.log(1-prob) + prob * Math.log(prob));
//              leftScore =inScore(in);
          }

          double rightScore;
          {
            csum[csum.length - 1] = weights[bf.index() * 2 + 1];
            ctotal[csum.length - 1] = total;
            final double prob = estimate(csum, ctotal);
            final AdditiveStatistics out = (AdditiveStatistics) loss.statsFactory().create();
            final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
            out.append(left);
            out.append(current.excluded);
            in.append(current.nonCriticalTotal);
            in.append(right);
            rightScore = (1 - prob) * outScore(out) + prob * inScore(in);// + 2 * ((1-prob) * Math.log(1-prob) + prob * Math.log(prob));
//              rightScore = score(out) +  score(in);
//              rightScore = inScore(in);
          }
          scores[bf.index()] = leftScore > rightScore ? rightScore : leftScore;
          isRight[bf.index()] = leftScore > rightScore;
        }
      });

//

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0)
        break;


      if ((scores[bestSplit] + 1e-9 >= currentScore))
        break;

      final BFGrid.Feature bestSplitBF = grid.bf(bestSplit);
      final boolean bestSplitMask = isRight[bestSplitBF.index()];

      BFOptimizationSubset outRegion = current.split(bestSplitBF, bestSplitMask);
      if (outRegion == null) {
        break;
      }
      outRegion = determenisticCurrent.split(bestSplitBF, bestSplitMask);
      if (outRegion == null) {
        break;
      }

      conditions.add(bestSplitBF);
      conditionSum.add(isRight[bestSplitBF.index()] ? weights[bestSplitBF.index() * 2 + 1] : weights[bestSplitBF.index() * 2]);
      conditionTotal.add(total);
      usedBF[bestSplitBF.index()] = true;
      mask.add(bestSplitMask);
      currentScore = scores[bestSplit];
    }


    final boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }

//
    final Region region = new Region(conditions, masks, 1, 0, -1, currentScore, conditions.size() > maxFailed ? maxFailed : 0);
    final Vec target = loss.target();
    double sum = 0;
    double outSum = 0;
    double weight = 0;

    for (int i = 0; i < bds.original().length(); ++i) {
      if (region.value(bds, i) == 1) {
        final double samplWeight = 1.0;// current.size() > 10 ? rand.nextPoisson(1.0) : 1.0;
        weight += samplWeight;
        sum += target.get(i) * samplWeight;
      } else {
        outSum += target.get(i);
      }
    }

    final double value = weight > 1 ? sum / weight : loss.bestIncrement(current.total());//loss.bestIncrement(inside);
//    double value = loss.bestIncrement(current.total());//loss.bestIncrement(inside);
    return new Region(conditions, masks, value, 0, -1, currentScore, conditions.size() > 1 ? maxFailed : 0);

  }

  private double estimate(final double[] sum, final double[] counts) {
    double p = 1;
    for (int i = 0; i < sum.length; ++i) {
      p *= sum[i] / counts[i];
    }
    return p;
//    double[] probs = MTA.bernoulliStein(sum, counts);
//    double p = 0;
//    for (int i = 0; i < sum.length; ++i) {
//      p += Math.log(probs[i]);
//    }
//    return Math.exp(p);
  }


  public static double weight(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.Stat) ((WeightedLoss.Stat) stats).inside).weight;
    }
    if (stats instanceof L2.Stat) {
      return ((L2.Stat) stats).weight;
    }
    return 0;
  }

  public static double sum(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.Stat) ((WeightedLoss.Stat) stats).inside).sum;
    }
    if (stats instanceof L2.Stat) {
      return ((L2.Stat) stats).sum;
    }
    return 0;
  }

  public static double sum2(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.Stat) ((WeightedLoss.Stat) stats).inside).sum2;
    }
    if (stats instanceof L2.Stat) {
      return ((L2.Stat) stats).sum2;
    }
    return 0;
  }

  public double score(final AdditiveStatistics stats) {
    final double sum = sum(stats);
    final double sum2 = sum2(stats);
    final double weight = weight(stats);
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
//    return weight > 1 ? (sum2 / (weight - 1) - sum * sum / (weight - 1) / (weight - 1)) : sum2;
//    return weight > 2 ? (sum2 / weight - sum * sum / weight / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;

//    final double n = stats.weight;
//    return n > 2 ? n*(n-2)/(n * n - 3 * n + 1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
//    return (stats.sum2 / (n-1) -stats.sum * stats.sum  / (n - 1) / (n - 1));
  }

  public double inScore(final AdditiveStatistics stats) {
    final double weight = weight(stats);
    if (weight < 5) {
      return Double.POSITIVE_INFINITY;
    }
    final double sum = sum(stats);
    final double sum2 = sum2(stats);
//    return weight > 2 ? (-sum * sum * weight / (weight-1) / (weight-1))  * (1 + 2 * Math.log(weight + 1)) : 0;
    return -sum * sum / (weight - 1) / (weight - 1);
//    return weight > 1 ? (sum2 / (weight - 1) - sum * sum / (weight - 1) / (weight - 1)) : sum2;
//    final double n = stats.weight;
//    return (weight - 2) /   (weight * weight - 3 * weight + 1) * (sum2 - sum * sum / weight);
//    return (sum2 /(weight-1) - sum * sum / weight / (weight-1));
//    return (stats.sum2 / (n-1) -stats.sum * stats.sum  / (n - 1) / (n - 1));
  }

  public double outScore(final AdditiveStatistics stats) {
    final double sum2 = sum2(stats);
    final double sum = sum(stats);
    final double weight = weight(stats);
    if (weight < 5) {
      return Double.POSITIVE_INFINITY;
    }
    return 0;
//    return weight > 1 ? (sum2 / (weight - 1) - sum * sum / (weight - 1) / (weight - 1)) : sum2;
//    return sum2 / (weight - 1) ;

//    return weight > 1 ? sum2 / (weight - 1) : sum2;
//
//    return weight > 1 ? sum2 / (weight - 1) : sum2;
//    final double n = stats.weight;
//    return n > 2 ? n*(n-2)/(n * n - 3 * n + 1) * (stats.sum2 - stats.sum * stats.sum / n) : stats.sum2;
//    return (stats.sum2 / (n-1) -stats.sum * stats.sum  / (n - 1) / (n - 1));
  }

  public double logScore(final AdditiveStatistics stats) {
    final double weight = weight(stats);
    final double sum = sum(stats);
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }
}
