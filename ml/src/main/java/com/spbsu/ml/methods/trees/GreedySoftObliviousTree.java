package com.spbsu.ml.methods.trees;

import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.softBorders.dataSet.SoftDataSet;
import com.spbsu.ml.data.softBorders.dataSet.SoftGrid;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.SoftObliviousTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/**
 * User: noxoomo
 */
public class GreedySoftObliviousTree<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final int depth;
  private final SoftGrid grid;

  public GreedySoftObliviousTree(final SoftGrid grid,
                                 final int depth) {
    this.grid = grid;
    this.depth = depth;
  }

  @Override
  public SoftObliviousTree fit(final VecDataSet ds,
                               final Loss loss) {

    final List<SoftGrid.SoftRow.BinFeature> conditions = new ArrayList<>(depth);
    final SoftDataSet softDataSet =  ds.cache()
            .cache(Binarize.class, VecDataSet.class)
            .softBinarize(grid);

    double currentScore = Double.POSITIVE_INFINITY;
    final double[] scores = new double[grid.binFeatureCount()];

    List<SoftBinFeaturesOptimizationSubset> leaves = new ArrayList<>(1 << depth);
    leaves.add(new SoftBinFeaturesOptimizationSubset(softDataSet, loss, ArrayTools.sequence(0, ds.length())));

    for (int level = 0; level < depth; level++) {
      Arrays.fill(scores, 0.);
      for (final SoftBinFeaturesOptimizationSubset leaf : leaves) {
        leaf.visitAllSplits((bf, left, right) -> scores[bf.bfIndex] += loss.score(left) + loss.score(right));
      }
      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;
      final SoftGrid.SoftRow.BinFeature bf = grid.bf(bestSplit);
      final List<SoftBinFeaturesOptimizationSubset> next = new ArrayList<>(leaves.size() * 2);
      final ListIterator<SoftBinFeaturesOptimizationSubset> iter = leaves.listIterator();
      while (iter.hasNext()) {
        final SoftBinFeaturesOptimizationSubset subset = iter.next();
        next.add(subset);
        next.add(subset.split(bf));
      }
      conditions.add(bf);
      leaves = next;
      currentScore = scores[bestSplit];
    }

    final double[] step = new double[leaves.size()];
    for (int i = 0; i < step.length; i++) {
      step[i] = loss.bestIncrement(leaves.get(i).total());
    }
    final SoftGrid.SoftRow.BinFeature[] binFeatures = conditions.toArray(new SoftGrid.SoftRow.BinFeature[conditions.size()]);
    return new SoftObliviousTree(binFeatures, step);
  }
}
