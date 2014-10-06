package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;


  public GreedyTDRegion(BFGrid grid) {
    this.grid = grid;
  }


  @Override
  public Region fit(VecDataSet learn, final Loss loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final List<Boolean> mask = new ArrayList<>();
    double currentScore = Double.POSITIVE_INFINITY;
    final AdditiveStatistics excluded = (AdditiveStatistics) loss.statsFactory().create();
    BFStochasticOptimizationSubset current;
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    current = new BFStochasticOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.length()));

    final double[] scores = new double[grid.size()];
    final boolean[] used = new boolean[grid.size()];
    final boolean[] usedFeatures = new boolean[grid.rows()];
    while (true) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          if (used[bf.bfIndex]) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
//            scores[bf.bfIndex] = loss.value(excluded)+ Math.min(loss.score(left) + loss.value(right), loss.score(right) + loss.value(left));
            scores[bf.bfIndex] = Math.min(loss.score(left), loss.score(right));
          }
        }
      });
      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] + 1e-9 >= currentScore)
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean[] bestSplitMask = new boolean[1];
      current.visitSplit(bestSplitBF, new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
//          bestSplitMask[0] = loss.score(left) + loss.value(right) > loss.score(right) + loss.value(left);
          bestSplitMask[0] = loss.score(left) > loss.score(right);
        }
      });

      BFStochasticOptimizationSubset inRegion = current.split(bestSplitBF, bestSplitMask[0]);
      if (inRegion == null) {
        break;
      }

//      excluded.append(current.total());
      conditions.add(bestSplitBF);
      used[bestSplitBF.bfIndex] = true;
      mask.add(bestSplitMask[0]);
      current = inRegion;
      currentScore = scores[bestSplit];
    }
    boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }
//    return new Region(conditions, masks, loss.bestIncrement(current.total()), loss.bestIncrement(excluded), -1, currentScore);
    return new Region(conditions, masks, loss.bestIncrement(current.total()), -1, currentScore);
  }
}
