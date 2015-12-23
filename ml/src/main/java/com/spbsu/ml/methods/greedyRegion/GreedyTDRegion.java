package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.Region;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegion<Loss extends StatBasedLoss> extends RegionBasedOptimization<Loss> {
  protected final BFGrid grid;
  private final FastRandom rand = new FastRandom();
  private final double alpha;
  private final double beta;
  private final int maxFailed;

  public GreedyTDRegion(final BFGrid grid) {
    this(grid, 0.02, 0.5, 1);
  }

  public GreedyTDRegion(final BFGrid grid, final double alpha, final double beta, final int maxFailed) {
    this.grid = grid;
    this.alpha = alpha;
    this.beta = beta;
    this.maxFailed = maxFailed;
  }

  public GreedyTDRegion(final BFGrid grid, final double alpha, final double beta) {
    this(grid, alpha, beta, 1);
  }


  Pair<BFGrid.BinaryFeature[], boolean[]> initFit(final VecDataSet learn, final Loss loss) {
    final BFOptimizationSubset current;
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    current = new BFOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.length()));
    final double[] bestRowScores = new double[grid.rows()];
    for (int i = 0; i < bestRowScores.length; ++i) {
      bestRowScores[i] = Double.POSITIVE_INFINITY;
    }
    final BFGrid.BinaryFeature[] bestRowFeatures = new BFGrid.BinaryFeature[grid.rows()];
    final boolean[] masks = new boolean[grid.rows()];

    current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
      @Override
      public void accept(final BFGrid.BinaryFeature bf, final AdditiveStatistics left, final AdditiveStatistics right) {
        final double leftScore = score(left);
        final double rightScore = score(right);
        final double bestScore = leftScore > rightScore ? rightScore : leftScore;

        if (bestScore < bestRowScores[bf.findex]) {
          bestRowScores[bf.findex] = bestScore;
          masks[bf.findex] = leftScore > rightScore;
          bestRowFeatures[bf.findex] = bf;
        }
      }
    });


    final boolean[] resultMasks = new boolean[maxFailed];
    final BFGrid.BinaryFeature[] resultFeatures = new BFGrid.BinaryFeature[maxFailed];

    for (int i = 0; i < maxFailed; ) {
      final boolean[] used = new boolean[bestRowScores.length];
      final int index = rand.nextInt(bestRowScores.length);
      if (bestRowScores[index] < Double.POSITIVE_INFINITY && !used[index]) {
        used[index] = true;
        final BFGrid.BinaryFeature feature = bestRowFeatures[index];
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
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final boolean[] usedBF = new boolean[grid.size()];
    final List<Boolean> mask = new ArrayList<>();
    final Pair<BFGrid.BinaryFeature[], boolean[]> init = initFit(learn, loss);
    for (int i = 0; i < init.first.length; ++i) {
      conditions.add(init.first[i]);
      usedBF[init.first[i].bfIndex] = true;
      mask.add(init.second[i]);
    }
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double currentScore = Double.POSITIVE_INFINITY;
    final BFWeakConditionsStochasticOptimizationRegion current =
            new BFWeakConditionsStochasticOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.first, init.second, maxFailed);
    current.alpha = alpha;
    current.beta = beta;
    final boolean[] isRight = new boolean[grid.size()];
    final double[] scores = new double[grid.size()];

    while (true) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(final BFGrid.BinaryFeature bf, final AdditiveStatistics left, final AdditiveStatistics right) {
          if (usedBF[bf.bfIndex]) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
            final double leftScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(left);
              leftScore = loss.score(in);
            }

            final double rightScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(right);
              rightScore = loss.score(in);
            }
            scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
            isRight[bf.bfIndex] = leftScore > rightScore;
          }
        }
      });

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0)
        break;


      if ((scores[bestSplit] + 1e-9 >= currentScore))
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean bestSplitMask = isRight[bestSplitBF.bfIndex];

      final BFOptimizationSubset outRegion = current.split(bestSplitBF, bestSplitMask);
      if (outRegion == null) {
        break;
      }

      conditions.add(bestSplitBF);
      usedBF[bestSplitBF.bfIndex] = true;
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
    double weight = 0;

    for (int i = 0; i < bds.original().length(); ++i) {
      if (region.contains(bds, i)) {
        final double samplWeight = 1.0;// current.size() > 10 ? rand.nextPoisson(1.0) : 1.0;
        weight += samplWeight;
        sum += target.get(i) * samplWeight;
      }
    }

    final double value = weight > 1 ? sum / weight : loss.bestIncrement(current.total());
    return new Region(conditions, masks, value, 0, -1, currentScore, conditions.size() > 1 ? maxFailed : 0);
  }


  public RegionStats findRegion(final VecDataSet learn, final Loss loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final boolean[] usedBF = new boolean[grid.size()];
    final List<Boolean> mask = new ArrayList<>();
    final Pair<BFGrid.BinaryFeature[], boolean[]> init = initFit(learn, loss);
    for (int i = 0; i < init.first.length; ++i) {
      conditions.add(init.first[i]);
      mask.add(init.second[i]);
    }
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double currentScore = Double.POSITIVE_INFINITY;
    final BFWeakConditionsStochasticOptimizationRegion current =
            new BFWeakConditionsStochasticOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.first, init.second, maxFailed);
    current.alpha = alpha;
    current.beta = beta;
    final boolean[] isRight = new boolean[grid.size()];
    final double[] scores = new double[grid.size()];

    while (true) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(final BFGrid.BinaryFeature bf, final AdditiveStatistics left, final AdditiveStatistics right) {
          if (usedBF[bf.bfIndex]) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
            final double leftScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(left);
              leftScore = loss.score(in);
            }

            final double rightScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(right);
              rightScore = loss.score(in);
            }
            scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
            isRight[bf.bfIndex] = leftScore > rightScore;
          }
        }
      });

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0)
        break;


      if ((scores[bestSplit] + 1e-9 >= currentScore))
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean bestSplitMask = isRight[bestSplitBF.bfIndex];

      final BFOptimizationSubset outRegion = current.split(bestSplitBF, bestSplitMask);
      if (outRegion == null) {
        break;
      }
      conditions.add(bestSplitBF);
      usedBF[bestSplitBF.bfIndex] = true;
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
    final TDoubleArrayList sample = new TDoubleArrayList();

    for (int i = 0; i < bds.original().length(); ++i) {
      if (region.contains(bds, i)) {
        sample.add(target.get(i));
      }
    }
    if (sample.size() == 0) {
      sample.add(0);
    }
    return new RegionStats(conditions, masks, sample, conditions.size() > 1 ? maxFailed : 0);
  }


  class RegionStats {
    final List<BFGrid.BinaryFeature> conditions;
    final boolean[] mask;
    final TDoubleArrayList inside;
    final int maxFailed;

    RegionStats(final List<BFGrid.BinaryFeature> conditions, final boolean[] mask, final TDoubleArrayList inside, final int maxFailed) {
      this.conditions = conditions;
      this.mask = mask;
      this.inside = inside;
      this.maxFailed = maxFailed;
    }
  }

  public double score(final AdditiveStatistics stats) {
    final double sum = sum(stats);
    final double weight = weight(stats);
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }


}
