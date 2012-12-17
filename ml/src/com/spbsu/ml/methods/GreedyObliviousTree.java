package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.Histogram;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.loss.LossFunction;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.TDoubleDoubleProcedure;
import gnu.trove.TIntArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousTree extends GreedyTDRegion {
  private final int depth;

  public GreedyObliviousTree(Random rng, DataSet ds, BFGrid grid, int depth) {
    super(rng, ds, grid, 1./3, 0);
    this.depth = depth;
  }

  @Override
  public ObliviousTree fit(DataSet ds, LossFunction loss) {
    assert loss instanceof L2Loss;
    List<int[]> split = new ArrayList<int[]>();
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();

    double[] totals = new double[1 << (depth + 1)];
    double[] weights = new double[1 << (depth + 1)];
    double currentScore = Double.MAX_VALUE;

    split.add(ds instanceof Bootstrap ? ((Bootstrap) ds).order() : ArrayTools.sequence(0, ds.power()));
    totals[0] = VecTools.sum(ds.target());
    weights[0] = split.get(0).length;
    for (int level = 0; level < depth; level++) {
      final Histogram[] histograms = new Histogram[split.size()];
      for (int i = 0; i < histograms.length; i++) {
        histograms[i] = bds.buildHistogram(target, split.get(i));
      }
      final BestBFFinder finder = new BestBFFinder(totals, weights, conditions.size() + 1);
      for (int bf = 0; bf < grid.size(); bf++, finder.advance()) {
        for (int fold = 0; fold < histograms.length; fold++)
          histograms[fold].process(bf, finder);
      }
      if (finder.bestScore > currentScore)
        break;
      final int bestSplit = finder.bestSplit();
      List<int[]> nextSplit = new ArrayList<int[]>(split.size() << 1);
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final byte[] bins = bds.bins(bestSplitBF.findex);
      int binNo = bestSplitBF.binNo;
      final TIntArrayList left = new TIntArrayList();
      final TIntArrayList right = new TIntArrayList();
      for (int s = 0; s < split.size(); s++) {
        double totalLeft = 0;
        double totalRight = 0;
        double totalWLeft = 0;
        double totalWRight = 0;
        left.clear();
        right.clear();
        int[] indices = split.get(s);
        for (int t = 0; t < indices.length; t++) {
          final int index = indices[t];
          if (bins[index] > binNo) {
            right.add(index);
            totalWRight++;
            totalRight += target.get(index);
          }
          else {
            left.add(index);
            totalWLeft++;
            totalLeft += target.get(index);
          }
        }
        totals[s * 2] = totalLeft;
        weights[s * 2] = totalWLeft;
        totals[s * 2 + 1] = totalRight;
        weights[s * 2 + 1] = totalWRight;
        nextSplit.add(left.toNativeArray());
        nextSplit.add(right.toNativeArray());
      }
      conditions.add(bestSplitBF);
      split = nextSplit;
      currentScore = finder.bestScore;
    }

    for (int i = 0; i < weights.length; i++)
      if (weights[i] > 2)
        totals[i] /= weights[i];
      else
        totals[i] = 0;
    return new ObliviousTree(conditions, totals, weights, currentScore);
  }

  private class BestBFFinder implements TDoubleDoubleProcedure {
    double score = 0;
    int fold = 0;
    double bestScore = Double.MAX_VALUE;
    int bestFeature = -1;

    int bfIndex = 0;

    final double[] totals;
    final double[] totalWeights;
    final int complexity;

    BestBFFinder(double[] totals, double[] totalWeights, int complexity) {
      this.totals = totals;
      this.totalWeights = totalWeights;
      this.complexity = complexity;
    }

    @Override
    public boolean execute(double weight, double sum) {
      double rightScore = score(this.totalWeights[fold] - weight, totals[fold] - sum, complexity);
      double leftScore = score(weight, sum, complexity);
      score += rightScore + leftScore;
      fold++;
      return true;
    }

    public void advance() {
      if (bestScore > score) {
        bestScore = score;
        bestFeature = bfIndex;
      }
      fold = 0;
      score = 0;
      bfIndex++;
    }

    public int bestSplit() {
      return bestFeature;
    }
  }
}
