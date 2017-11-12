package com.expleague.ml.methods.trees;

import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.ObliviousTree;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousTree<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final int depth;
  public final BFGrid grid;

  public GreedyObliviousTree(final BFGrid grid, final int depth) {
    this.grid = grid;
    this.depth = depth;
  }

  @Override
  public ObliviousTree fit(final VecDataSet ds,
                           final Loss loss) {
    Pair<List<BFOptimizationSubset>, List<BFGrid.BinaryFeature>> result = findBestSubsets(ds,loss);
    List<BFOptimizationSubset> leaves = result.getFirst();
    List<BFGrid.BinaryFeature> conditions = result.getSecond();
    final double[] step = new double[leaves.size()];
    final double[] based = new double[leaves.size()];
    for (int i = 0; i < step.length; i++) {
      step[i] = loss.bestIncrement(leaves.get(i).total());
      based[i] = leaves.get(i).size();
    }
    return new ObliviousTree(conditions, step, based);
  }

  //decomposition for oblivious tree with non-constant functions in leaves
  public final Pair<List<BFOptimizationSubset>,List<BFGrid.BinaryFeature>> findBestSubsets(final VecDataSet ds, final Loss loss) {
    List<BFOptimizationSubset> leaves = new ArrayList<BFOptimizationSubset>(1 << depth);
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedDataSet bds =  ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    leaves.add(new BFOptimizationSubset(bds, loss, learnPoints(loss, ds)));

    final double[] scores = new double[grid.size()];
    for (int level = 0; level < depth; level++) {
      Arrays.fill(scores, 0.);
      for (final BFOptimizationSubset leaf : leaves) {
        leaf.visitAllSplits((bf, left, right) -> scores[bf.bfIndex] += loss.score(left) + loss.score(right));
      }
      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final List<BFOptimizationSubset> next = new ArrayList<>(leaves.size() * 2);
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
    return new Pair<>(leaves, conditions);
  }

  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }
}
