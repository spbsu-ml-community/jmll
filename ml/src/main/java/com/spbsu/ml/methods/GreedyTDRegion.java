package com.spbsu.ml.methods;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegion<O extends StatBasedLoss> implements Optimization<O> {
  protected final BFGrid grid;

  public GreedyTDRegion(BFGrid grid) {
    this.grid = grid;
  }

  @Override
  public Region fit(DataSet learn, final O loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(100);
    final List<Boolean> mask = new ArrayList<Boolean>();
    double currentScore = Double.POSITIVE_INFINITY;
    final AdditiveStatistics excluded = (AdditiveStatistics)loss.statsFactory().create();
    BFOptimizationSubset current;
    final BinarizedDataSet bds = learn.cache(Binarize.class).binarize(grid);
    current = new BFOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.power()));


    final double[] scores = new double[grid.size()];
    while (true) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          scores[bf.bfIndex] = loss.value(excluded) +
                  Math.min(loss.score(left) + loss.value(right),
                           loss.value(left) + loss.score(right));
        }
      });
      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean[] isRight = new boolean[1];
      current.visitSplit(bestSplitBF, new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          isRight[0] = (loss.score(left) + loss.value(right) > loss.value(left) + loss.score(right));
        }
      });
      if (isRight[0]) {
        BFOptimizationSubset right = current.split(bestSplitBF);
        excluded.append(current.total());
        current = right;
        mask.add(true);
      }
      else {
        excluded.append(current.split(bestSplitBF).total());
        mask.add(false);
      }
      conditions.add(bestSplitBF);
      currentScore = scores[bestSplit];
    }
    boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }
    return new Region(conditions, masks, loss.bestIncrement(current.total()), -1, currentScore);
  }
}
