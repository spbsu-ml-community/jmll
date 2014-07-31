package com.spbsu.ml.DynamicGrid.Trees;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.DynamicGrid.AggregateDynamic;
import com.spbsu.ml.DynamicGrid.Impl.BFDynamicGrid;
import com.spbsu.ml.DynamicGrid.Impl.BinarizedDynamicDataSet;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Models.ObliviousTreeDynamicBin;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/**
 * Created by noxoomo on 22/07/14.
 */
public class GreedyObliviousTreeDynamic2<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final int depth;
  private DynamicGrid grid;
  private boolean growGrid = true;
  private final int minSplits;
  private final double lambda;
  private double eps = 1e-4;


  public GreedyObliviousTreeDynamic2(DynamicGrid grid, int depth) {
    this.depth = depth;
    this.grid = grid;
    minSplits = 1;
    lambda = 0.2;
  }

  public GreedyObliviousTreeDynamic2(VecDataSet ds, int depth) {
    this(ds, depth, 0, 1);
  }

  public GreedyObliviousTreeDynamic2(VecDataSet ds, int depth, double lambda) {
    this(ds, depth, lambda, 1);
  }

  public GreedyObliviousTreeDynamic2(VecDataSet ds, int depth, double lambda, int minSplits) {
    this.minSplits = minSplits;
    this.depth = depth;
    this.lambda = lambda;

    this.grid = new BFDynamicGrid(ds, minSplits);
  }


  public void stopGrowing() {
    this.growGrid = false;
  }

  public ObliviousTreeDynamicBin fit(VecDataSet ds, final Loss loss) {
    BinarizedDynamicDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    List<BFDynamicOptimizationSubset> leaves = new ArrayList<>(1 << depth);
    TIntArrayList nonActiveF = new TIntArrayList(grid.rows());
    TIntArrayList nonActiveBin = new TIntArrayList(grid.rows());
    final List<BinaryFeature> conditions = new ArrayList<>(depth);
    final double[][] scores = new double[grid.rows()][];
    for (int i = 0; i < scores.length; ++i) {
      scores[i] = new double[0];
    }

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


      for (BFDynamicOptimizationSubset leaf : leaves) {
        leaf.visitAllSplits(new AggregateDynamic.SplitVisitor<AdditiveStatistics>() {
          @Override
          public void accept(BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
            double leftScore = loss.score(left);
            double rightScore = loss.score(right);
            scores[bf.fIndex()][bf.binNo()] += leftScore + rightScore;
          }
        });
      }

      int bestSplitF = -1;
      int bestSplitBin = -1;
      double bestSplitScore = Double.POSITIVE_INFINITY;


      for (int f = 0; f < scores.length; ++f) {

        int candidateIndex = -1;

        double bestFeatureScore = Double.POSITIVE_INFINITY;
        int bestBinIndex = -1;


        for (int bin = 0; bin < scores[f].length; ++bin) {
          BinaryFeature bf = grid.bf(f, bin);
          if (bf.isActive()) {
            if (scores[f][bin] < bestFeatureScore) {
              bestBinIndex = bin;
              bestFeatureScore = scores[f][bin];
            }
          } else {
            candidateIndex = bin;
          }
        }

        if (candidateIndex != -1 && growGrid) {
          BinaryFeature candidate = grid.bf(f, candidateIndex);

          double score = bestFeatureScore - scores[f][candidateIndex];
          double reg = candidate.regularization();
          score -= lambda * reg;
          if (score > 0) {
            bds.queueSplit(candidate);
            bestBinIndex = candidateIndex;
          }
        }
        if (bestBinIndex == -1)
          continue;

        if (scores[f][bestBinIndex] < bestSplitScore) {
          bestSplitBin = bestBinIndex;
          bestSplitF = f;
          bestSplitScore = scores[f][bestBinIndex];
        }
      }


      //tree growing continue
      if (bestSplitF < 0 || scores[bestSplitF][bestSplitBin] >= currentScore) {
        if (growGrid) {
          bds.acceptQueue(leaves);
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
        bds.acceptQueue(leaves);
      }
    }

//            updated = false;
    double[] values = new double[leaves.size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = loss.bestIncrement(leaves.get(i).total());
    }
    for (BinaryFeature bf : conditions) {
      bf.use();
    }
    return new ObliviousTreeDynamicBin(conditions, values);
  }


  public int[] hist() {
    return grid.hist();
  }
}


