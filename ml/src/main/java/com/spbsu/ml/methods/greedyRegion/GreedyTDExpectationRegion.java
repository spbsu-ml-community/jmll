package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDExpectationRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  private final int maxFailed = 1;

  public GreedyTDExpectationRegion(BFGrid grid) {
    this.grid = grid;
  }

  FastRandom rand = new FastRandom();

  StatBasedLoss bootstrap(Loss loss, double[] prob) {
    int[] poissonWeights = new int[loss.xdim()];
    for (int i = 0; i < loss.xdim(); i++) {
      poissonWeights[i] = rand.nextPoisson(prob[i] * prob.length);
    }
    return new WeightedLoss<>(loss, poissonWeights);
  }


  @Override
  public Region fit(VecDataSet learn, final Loss loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final List<Boolean> mask = new ArrayList<>();
    double currentScore = Double.POSITIVE_INFINITY;
    final AdditiveStatistics excluded = (AdditiveStatistics) loss.statsFactory().create();
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double probs[] = new double[learn.length()];
    Arrays.fill(probs, 1.0);
    BFExpectationOptimizationSubset subset = new BFExpectationOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.length()), probs);

    final double[] scores = new double[grid.size()];
    final boolean[] used = new boolean[grid.size()];
    final short[] usedFeatures = new short[grid.rows()];

    while (true) {
      subset.visitAllSplits(new ExpectationAggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          if (usedFeatures[bf.findex] != 0) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
            AdditiveStatistics best = loss.score(left) > loss.score(right) ? right : left;
//            scores[bf.bfIndex] = loss.value(excluded)+ Math.min(loss.score(left) + loss.value(right), loss.score(right) + loss.value(left));
            scores[bf.bfIndex] = loss.score(best);
          }
        }
      });
      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] + 1e-9 >= currentScore)
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean[] bestSplitMask = new boolean[1];
      subset.visitSplit(bestSplitBF, new ExpectationAggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
//          bestSplitMask[0] = loss.score(left) + loss.value(right) > loss.score(right) + loss.value(left);
          bestSplitMask[0] = loss.score(left) > loss.score(right);
        }
      });


//      excluded.append(current.total());
      conditions.add(bestSplitBF);
      used[bestSplitBF.bfIndex] = true;
      usedFeatures[bestSplitBF.findex]++;
      mask.add(bestSplitMask[0]);
      subset.split(bestSplitBF, bestSplitMask[0]);
//      current = outRegion;
      currentScore = scores[bestSplit];
    }
    boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }

    Region region = new Region(conditions, masks, 1, 0, -1, currentScore, maxFailed);
    AdditiveStatistics inside = (AdditiveStatistics) loss.statsFactory().create();
    for (int i = 0; i < bds.original().length(); ++i) {
      if (region.value(bds, i) == 1) {
        inside.append(i, 1);
      }
    }
    double value = loss.bestIncrement(inside);
//    double value = loss.bestIncrement(current.total());
    return new Region(conditions, masks, value, 0, -1, currentScore, maxFailed);

//    return new Region(conditions, masks, loss.bestIncrement(subset.total()), 0, -1, currentScore,1);
//    return new Region(conditions, masks, loss.bestIncrement(current.total()), -1, currentScore);
  }
}
