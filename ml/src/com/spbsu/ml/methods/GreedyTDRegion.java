package com.spbsu.ml.methods;

import com.spbsu.commons.func.AdditiveGator;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.Model;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.StatBasedOracle;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegion<O extends StatBasedOracle> implements MLMethod<O> {
  private final Random rng;
  protected final BFGrid grid;
  protected final BinarizedDataSet bds;

  public GreedyTDRegion(Random rng, DataSet ds, BFGrid grid) {
    this.rng = rng;
    this.grid = grid;
    bds = new BinarizedDataSet(ds, grid);
  }

  @Override
  public Model fit(DataSet learn, final O loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(100);
    final List<Boolean> mask = new ArrayList<Boolean>();
    double currentScore = Double.POSITIVE_INFINITY;
    final AdditiveGator excluded = (AdditiveGator)loss.statsFactory().create();
    BFOptimizationSubset current;
    final BinarizedDataSet bds;
    if (learn instanceof Bootstrap) {
      final Bootstrap bs = (Bootstrap) learn;
      bds = bs.original().cache(Binarize.class).binarize(grid);
      current = new BFOptimizationSubset(bds, loss, bs.order());
    }
    else {
      bds = learn.cache(Binarize.class).binarize(grid);
      current = new BFOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.power()));
    }


    final double[] scores = new double[grid.size()];
    while (true) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveGator>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveGator left, AdditiveGator right) {
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
      current.visitSplit(bestSplitBF, new Aggregate.SplitVisitor<AdditiveGator>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveGator left, AdditiveGator right) {
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
    return new Region(conditions, masks, loss.gradient(current.total()), -1, currentScore);
  }
}
