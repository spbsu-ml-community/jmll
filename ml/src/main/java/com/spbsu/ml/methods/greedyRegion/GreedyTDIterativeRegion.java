package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDIterativeRegion<Loss extends StatBasedLoss> extends RegionBasedOptimization<Loss> {
  protected final BFGrid grid;
  private final FastRandom rand = new FastRandom();
  private final double alpha;
  private final double beta;

  public GreedyTDIterativeRegion(final BFGrid grid) {
    this(grid, 0.7, 0.5);
  }

  public GreedyTDIterativeRegion(final BFGrid grid, final double alpha, final double beta) {
    this.grid = grid;
    this.alpha = alpha;
    this.beta = beta;
  }


  @Override
  public Region fit(final VecDataSet learn, final Loss loss) {
    Region current = new Region(new ArrayList<BFGrid.BinaryFeature>(), null, 0, 0, 0, Double.POSITIVE_INFINITY, -1);
    while (true) {
      Region next = fitWeak(learn, loss, current, current.maxFailed + 1);
      if (next.score + 1e-9f >= current.score)
        return current;
      current = next;
    }
  }


  public Region fitWeak(final VecDataSet learn, final Loss loss, final Region init, final int maxFailed) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final boolean[] usedBF = new boolean[grid.size()];
    final List<Boolean> mask = new ArrayList<>();
    for (int i = 0; i < init.features.length; ++i) {
      conditions.add(init.features[i]);
      usedBF[init.features[i].bfIndex] = true;
      mask.add(init.mask[i]);
    }
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    final BFWeakConditionsOptimizationRegion current =
            new BFWeakConditionsOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.features, init.mask, maxFailed);
//    final BFWeakConditionsStochasticOptimizationRegion current =
//            new BFWeakConditionsStochasticOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.first, init.second, maxFailed);
//    current.alpha = alpha;
//    current.beta = beta;
    AdditiveStatistics currentInside = (AdditiveStatistics) loss.statsFactory().create();
    AdditiveStatistics currentCritical = (AdditiveStatistics) loss.statsFactory().create();
    AdditiveStatistics currentOutside = (AdditiveStatistics) loss.statsFactory().create();
    currentInside.append(current.total());
    currentOutside.append(current.excluded);
    currentCritical.append(currentInside);
    currentCritical.remove(current.nonCriticalTotal);
    final boolean[] isRight = new boolean[grid.size()];
    final double[] scores = new double[grid.size()];
//    double reg = (1 + 2*(Math.log(weight(currentInside) + 1) + Math.log(weight(currentOutside) + 1)));
//    reg /= (2 + 2*(maxFailed + conditions.size()));
//    double currentScore = loss.score(currentInside) * reg;
    double currentScore = loss.score(currentInside) * (1 +  Math.log(weight(currentInside) + 1) + (conditions.size() + maxFailed) * Math.log(alpha));
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
              final AdditiveStatistics out = (AdditiveStatistics) loss.statsFactory().create();
              out.append(current.excluded);
              out.append(right);
              double reg = 1 + (Math.log(weight(in) + 1)) + (conditions.size() + maxFailed + 1) * Math.log(alpha);
//              reg /= (1 + maxFailed);
//              / Math.log(2 + maxFailed + conditions.size())
//              leftScore = (loss.score(in) + loss.score(out)) / Math.log(2 + maxFailed + conditions.size());
//              leftScore = (loss.score(in)) * reg
              leftScore = loss.score(in) * reg;
            }

            final double rightScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(right);
              final AdditiveStatistics out = (AdditiveStatistics) loss.statsFactory().create();
              out.append(current.excluded);
              out.append(left);
//              reg /= (1 + 0.5maxFailed);
//              / Math.log(2 + maxFailed + conditions.size())
//              rightScore = (loss.score(in) + loss.score(out)) / Math.log(2 + maxFailed + conditions.size());
//              rightScore = (loss.score(in)) * reg;
              double reg = 1 +  (Math.log(weight(in) + 1)) + (conditions.size() + maxFailed + 1) * Math.log(alpha);
              rightScore = loss.score(in) * reg;
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
      currentInside = (AdditiveStatistics) loss.statsFactory().create();
      currentInside.append(current.total());
      currentOutside = (AdditiveStatistics) loss.statsFactory().create();
      currentOutside.append(current.excluded);
    }


    final boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }
    final Region region = new Region(conditions, masks,
//            loss.bestIncrement(currentInside), loss.bestIncrement(currentOutside), -1, currentScore, maxFailed);
            loss.bestIncrement(currentInside), 0, -1, currentScore, maxFailed);
    return region;
  }


}
