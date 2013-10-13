package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.Histogram;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.methods.GreedyTDRegion;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousRegressionTree extends GreedyTDRegion {
  private final int depth;

  public GreedyObliviousRegressionTree(Random rng, DataSet ds, BFGrid grid, int depth) {
    super(rng, ds, grid, 1./3, 0);
    this.depth = depth;
  }

  @Override
  public ObliviousTree fit(DataSet ds, Oracle1 loss, Vec start) {
    if(!(loss instanceof L2Loss))
      throw new IllegalArgumentException("L2 loss supported only");
    List<int[]> split = new ArrayList<int[]>();
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();
    final Vec point = VecTools.copy(start);

    double[] totals = new double[1 << depth];
    double[] totals2 = new double[1 << depth];
    double[] weights = new double[1 << depth];
    double currentScore = Double.POSITIVE_INFINITY;

    split.add(ds instanceof Bootstrap ? ((Bootstrap) ds).order() : ArrayTools.sequence(0, ds.power()));
    totals[0] = VecTools.sum(ds.target());
    totals2[0] = VecTools.sum2(ds.target());
    weights[0] = split.get(0).length;

    final double[] scores = new double[grid.size()];
    for (int level = 0; level < depth; level++) {
      final int complexity = conditions.size() + 1;
      Arrays.fill(scores, 0.);
      for (int i = 0; i < split.size(); i++) {
        final Histogram h = bds.buildHistogram(target, point, split.get(i));
        final double total = totals[i];
        final double total2 = totals2[i];
        final double totalWeight = weights[i];
        h.score(scores, new Histogram.Judge() {
          @Override
          public double score(double sum, double sum2, double weight, int bf) {
            double leftScore = scoreInner(sum, sum2, weight);
            double rightScore = scoreInner((total - sum), total2 - sum2, totalWeight - weight);
            return rightScore + leftScore;
          }

          private double scoreInner(double sum, double sum2, double weight) {
//            return GreedyObliviousRegressionTree.this.score(weight, sum, 1);
            if (weight > 1.) {
              return weight / (weight - 1) / (weight - 1) * (weight * sum2 - sum * sum);
            }
            else return sum2;
//            return (sum2 - (weight > MathTools.EPSILON ? sum * sum / weight : 0));
          }
        });
      }
      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] >= currentScore)
        break;
      List<int[]> nextSplit = new ArrayList<int[]>(split.size() << 1);
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final byte[] bins = bds.bins(bestSplitBF.findex);
      int binNo = bestSplitBF.binNo;
      final TIntArrayList left = new TIntArrayList(point.dim());
      final TIntArrayList right = new TIntArrayList();
      Arrays.fill(totals, 0.);
      Arrays.fill(totals2, 0.);
      for (int s = 0; s < split.size(); s++) {
        left.clear();
        right.clear();
        int rightI = s * 2 + 1, leftI = s * 2;
        int[] indices = split.get(s);
        for (int t = 0; t < indices.length; t++) {
          final int index = indices[t];
          final double v = target.get(index);
          final boolean isRight = bins[index] > binNo;
          final int leaf = isRight ? rightI : leftI;
          (isRight ? right : left).add(index);
          totals[leaf] += v;
          totals2[leaf] += v * v;
        }
        weights[s * 2] = left.size();
        weights[s * 2 + 1] = right.size();
        nextSplit.add(left.toNativeArray());
        nextSplit.add(right.toNativeArray());
      }
      conditions.add(bestSplitBF);
      split = nextSplit;
      currentScore = scores[bestSplit];
    }

    for (int i = 0; i < weights.length; i++)
      if (weights[i] > 2)
        totals[i] /= weights[i];
      else
        totals[i] = 0;
    return new ObliviousTree(conditions, totals, weights);
  }
}
