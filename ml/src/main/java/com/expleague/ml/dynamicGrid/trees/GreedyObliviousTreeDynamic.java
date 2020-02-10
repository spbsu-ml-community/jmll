package com.expleague.ml.dynamicGrid.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.dynamicGrid.AggregateDynamic;
import com.expleague.ml.dynamicGrid.impl.BFDynamicGrid;
import com.expleague.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.expleague.ml.dynamicGrid.interfaces.BinaryFeature;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.methods.VecOptimization;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/**
 * Created by noxoomo on 22/07/14.
 */
public class GreedyObliviousTreeDynamic<Loss extends AdditiveLoss> extends VecOptimization.Stub<Loss> {
  private final int depth;
  private final DynamicGrid grid;
  private boolean growGrid = true;
  //  private final int minSplits;
  private final double lambda;
  private static final double eps = 1e-4;


  public GreedyObliviousTreeDynamic(final DynamicGrid grid, final int depth, final double lambda) {
    this.depth = depth;
    this.grid = grid;
//    minSplits = 1;
//    lambda = 1;
    this.lambda = lambda;
  }

  public GreedyObliviousTreeDynamic(final VecDataSet ds, final int depth) {
    this(ds, depth, 0, 1);
  }

  public GreedyObliviousTreeDynamic(final VecDataSet ds, final int depth, final double lambda) {
    this(ds, depth, lambda, 1);
  }

  public GreedyObliviousTreeDynamic(final VecDataSet ds, final int depth, final double lambda, final int minSplits) {
//    this.minSplits = minSplits;
    this.depth = depth;
    this.lambda = lambda;
    this.grid = new BFDynamicGrid(ds, minSplits);
  }


  public void stopGrowing() {
    this.growGrid = false;
  }

  @Override
  public ObliviousTreeDynamicBin fit(final VecDataSet ds, final Loss loss) {
    final BinarizedDynamicDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    List<BFDynamicOptimizationSubset> leaves = new ArrayList<>(1 << depth);
    final TIntArrayList nonActiveF = new TIntArrayList(grid.rows());
    final TIntArrayList nonActiveBin = new TIntArrayList(grid.rows());
    final List<BinaryFeature> conditions = new ArrayList<>(depth);
    final double[][] scores = new double[grid.rows()][];
    for (int i = 0; i < scores.length; ++i) {
      scores[i] = new double[0];
    }

    while (true) {
      boolean updated = false;
      leaves.clear();
      conditions.clear();
      leaves.add(new BFDynamicOptimizationSubset(bds, loss, ArrayTools.sequence(0, ds.length())));
      double currentScore = Double.POSITIVE_INFINITY;

      for (int level = 0; level < depth; level++) {
        for (int f = 0; f < scores.length; ++f) {
          if (scores[f].length != grid.row(f).size()) {
            scores[f] = new double[grid.row(f).size()];
          } else Arrays.fill(scores[f], 0);
        }


        for (final BFDynamicOptimizationSubset leaf : leaves) {
          leaf.visitAllSplits(new AggregateDynamic.SplitVisitor<AdditiveStatistics>() {
            @Override
            public void accept(final BinaryFeature bf, final AdditiveStatistics left, final AdditiveStatistics right) {
              final double leftScore = loss.score(left);
              final double rightScore = loss.score(right);
              scores[bf.fIndex()][bf.binNo()] += leftScore + rightScore;
            }
          });
        }

        int bestSplitF = -1;
        int bestSplitBin = -1;
        double bestSplitScore = Double.POSITIVE_INFINITY;

        int bestNonActiveSplitF = -1;
        int bestNonActiveSplitBin = -1;
        double bestNonActiveSplitScore = Double.POSITIVE_INFINITY;
        nonActiveF.clear();
        nonActiveBin.clear();
        for (int f = 0; f < scores.length; ++f) {
          for (int bin = 0; bin < scores[f].length; ++bin) {
            final BinaryFeature bf = grid.bf(f, bin);
            if (bf.isActive()) {
              if (bestSplitScore > scores[f][bin]) {
                bestSplitF = f;
                bestSplitBin = bin;
                bestSplitScore = scores[f][bin];
              }
            } else {
              nonActiveF.add(f);
              nonActiveBin.add(bin);
            }
          }
        }
        if (growGrid) {
          final double threshold = bestSplitScore < currentScore ? bestSplitScore : currentScore;
          for (int j = 0; j < nonActiveF.size(); ++j) {
            final int feature = nonActiveF.get(j);
            final int bin = nonActiveBin.get(j);
            final BinaryFeature bf = grid.bf(feature, bin);
            final double reg = lambda != 0 ? bf.regularization() : 0;
            final double score = threshold - scores[feature][bin] - lambda * reg;
            if (score > eps) {
              bds.queueSplit(bf);
              if (bestNonActiveSplitScore > scores[feature][bin]) {
                bestNonActiveSplitF = feature;
                bestNonActiveSplitBin = bin;
                bestNonActiveSplitScore = scores[feature][bin];
              }
            }
          }
        }

        if (bestNonActiveSplitScore <= bestSplitScore) {
          bestSplitF = bestNonActiveSplitF;
          bestSplitBin = bestNonActiveSplitBin;
        }


        //tree growing continue
        if (bestSplitF < 0 || scores[bestSplitF][bestSplitBin] >= currentScore) {
          if (growGrid) {
            if (bds.acceptQueue(leaves)) {
              updated = true;
            }
          }
          break;
        }
        final BinaryFeature bestSplitBF = grid.bf(bestSplitF, bestSplitBin);
        final List<BFDynamicOptimizationSubset> next = new ArrayList<>(leaves.size() * 2);
        final ListIterator<BFDynamicOptimizationSubset> iter = leaves.listIterator();
        while (iter.hasNext()) {
          final BFDynamicOptimizationSubset subset = iter.next();
          next.add(subset);
          next.add(subset.split(bestSplitBF));
        }
        conditions.add(bestSplitBF);
        leaves = next;
        currentScore = scores[bestSplitF][bestSplitBin];
        if (growGrid) {
          if (bds.acceptQueue(leaves)) {
            updated = true;
          }
        }
      }

//      updated = false;
      if (!updated) {
        final double[] values = new double[leaves.size()];
        for (int i = 0; i < values.length; i++) {
          int lastFeature = conditions.get(conditions.size() - 1).fIndex();
          values[i] = loss.bestIncrement(leaves.get(i).total(lastFeature));
        }
//        for (Feature bf : conditions) {
//          bf.use();
//        }
        return new ObliviousTreeDynamicBin(conditions, values);
      }
    }

  }

  public int[] hist() {
    return grid.hist();
  }
}
