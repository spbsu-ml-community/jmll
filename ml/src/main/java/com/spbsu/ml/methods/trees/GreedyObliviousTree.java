package com.spbsu.ml.methods.trees;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.VectorizedRealTargetDataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousTree<Loss extends StatBasedLoss> implements VecOptimization<Loss> {
  private final int depth;
  private final BFGrid grid;

  public GreedyObliviousTree(BFGrid grid, int depth) {
    this.grid = grid;
    this.depth = depth;
  }

  @Override
  public ObliviousTree fit(VectorizedRealTargetDataSet<?> ds, final Loss loss) {
    List<BFOptimizationSubset> leaves = new ArrayList<BFOptimizationSubset>(1 << depth);
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedDataSet bds;
    bds = ds.cache(Binarize.class).binarize(grid);
    leaves.add(new BFOptimizationSubset(bds, loss, ArrayTools.sequence(0, ds.length())));

    final double[] scores = new double[grid.size()];
    for (int level = 0; level < depth; level++) {
      Arrays.fill(scores, 0.);
      for (BFOptimizationSubset leaf : leaves) {
        leaf.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
          @Override
          public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
            scores[bf.bfIndex] += loss.score(left) + loss.score(right);
          }
        });
      }
      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final List<BFOptimizationSubset> next = new ArrayList<BFOptimizationSubset>(leaves.size() * 2);
      final ListIterator<BFOptimizationSubset> iter = leaves.listIterator();
      while (iter.hasNext()) {
        final BFOptimizationSubset subset = iter.next();
        next.add(subset);
        next.add(subset.split(bestSplitBF));
      }
      conditions.add(bestSplitBF);
      leaves = next;
      currentScore = scores[bestSplit];
    }
    double[] step = new double[leaves.size()];
    double[] based = new double[leaves.size()];
    for (int i = 0; i < step.length; i++) {
      step[i] = loss.bestIncrement(leaves.get(i).total());
    }
    return new ObliviousTree(conditions, step, based);
  }
}
