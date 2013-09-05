package com.spbsu.ml.methods;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.Aggregator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.LogLikelihoodSigmoid;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.TIntArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousClassificationTree implements MLMethodOrder1 {
  private final Random rng;
  private final BinarizedDataSet ds;
  private final int depth;
  private BFGrid grid;

  public GreedyObliviousClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
    this.rng = rng;
    this.depth = depth;
    this.ds = new BinarizedDataSet(ds, grid);
    this.grid = grid;
  }

  @Override
  public Model fit(DataSet learn, Oracle1 loss) {
    return fit(learn, loss, new ArrayVec(learn.power()));
  }

  @Override
  public ObliviousTree fit(DataSet ds, Oracle1 loss, Vec point) {
    if(!(loss instanceof LogLikelihoodSigmoid))
      throw new IllegalArgumentException("Log likelihood with sigmoid probability function supported only");
    List<int[]> split = new ArrayList<int[]>();
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();

    LLAggregator[] agg = new LLAggregator[0];
    int bestSplit = -1;

    split.add(ds instanceof Bootstrap ? ((Bootstrap) ds).order() : ArrayTools.sequence(0, ds.power()));
    for (int level = 0; level < depth; level++) {
      agg = new LLAggregator[split.size()];
      for (int i = 0; i < agg.length; i++) {
        agg[i] = new LLAggregator();
        this.ds.aggregate(agg[i], target, point, split.get(i));
      }
      double[] scores = new double[grid.size()];
      for (int i = 0; i < agg.length; i++) {
        agg[i].score(scores);
      }
      bestSplit = ArrayTools.max(scores);
      List<int[]> nextSplit = new ArrayList<int[]>(split.size() << 1);
      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final byte[] bins = this.ds.bins(bestSplitBF.findex);
      int binNo = bestSplitBF.binNo;
      final TIntArrayList left = new TIntArrayList();
      final TIntArrayList right = new TIntArrayList();
      for (int s = 0; s < split.size(); s++) {
        left.clear();
        right.clear();
        int[] indices = split.get(s);
        for (int t = 0; t < indices.length; t++) {
          final int index = indices[t];
          if (bins[index] > binNo) {
            right.add(index);
          }
          else {
            left.add(index);
          }
        }
        nextSplit.add(left.toNativeArray());
        nextSplit.add(right.toNativeArray());
      }
      conditions.add(bestSplitBF);
      split = nextSplit;
    }

    double[] values = new double[split.size()];
    double[] weights = new double[split.size()];
    {
      for (int i = 0; i < weights.length; i++) {

        values[i] = i %2 == 0 ? agg[i/2].left(bestSplit) : agg[i/2].right(bestSplit);
        weights[i] = split.get(i).length;
      }
    }
    return new ObliviousTree(conditions, values, weights);
  }

  /** Key idea is to find \min_s \sum_i log \frac{1}{1 + e^{-(x_i + s})y_i}, where x_i -- current score, y_i \in \{-1,1\} -- category
   * for this we need to get solution for \sum_i \frac{y_i}{1 + e^{y_i(x_i + s}}. This equation is difficult to solve in closed form so
   * we use Taylor series approximation. For this we need to make substitution s = log(1-v) - log(1+v) so that Maclaurin series in terms of
   * v were limited.
   *
   * LLCounter is a class which calculates Maclaurin coefficients of original function and its first derivation.
   */
  private class LLCounter {
    public int good = 0;
    public int bad = 0;
    public int unknown = 0;
    public double[] maclaurinLL = new double[5];
    private double alpha = Double.NaN;

    public double alpha() {
      if (good == 0 || bad == 0)
        return 0;
      if (!Double.isNaN(alpha))
        return alpha;
      final double[] x = new double[3];

      int cnt = MathTools.cubic(x, maclaurinLL[4], maclaurinLL[3], maclaurinLL[2], maclaurinLL[1]);
      double y = 0.;
      double bestLL = maclaurinLL[0];
      for (int i = 0; i < cnt; i++) {
        if (score(x[i]) > bestLL) {
          y = x[i];
          bestLL = score(y);
        }
      }

      return alpha = log((1. - y) / (1. + y));
    }

    private double score(double x) {
      if (good == 0 || bad == 0)
        return maclaurinLL[0];
      return maclaurinLL[0] - 2 * maclaurinLL[1] * x - maclaurinLL[2] * x * x;
    }

    public double score() {
      double alpha = alpha();
      double x = (1 - exp(alpha))/(1 + exp(alpha));
      return score(x);
    }


    public void found(double current, double target, double weight) {
      final double b = target > 0 ? 1. : -1.;
      final double eab = exp(current*b);
      final double eabPlusOne = eab + 1;
      final double eabMinusOne = eab - 1;
      double denominator = eabPlusOne;
      maclaurinLL[0] += weight * log(eab/(1. + eab));
      maclaurinLL[1] += weight * b/denominator;
      denominator *= eabPlusOne;
      maclaurinLL[2] += weight * 2 * eab/denominator;
      denominator *= eabPlusOne;
      maclaurinLL[3] += weight * 2 * b * eab * eabMinusOne /denominator;
      denominator *= eabPlusOne;
      maclaurinLL[4] += weight * 2 * eab * eabMinusOne * eabMinusOne / denominator;
      if (b > 0)
        good++;
      else
        bad++;
    }

    public void append(LLCounter counter) {
      for (int i = 0; i < maclaurinLL.length; i++) {
        maclaurinLL[i] += counter.maclaurinLL[i];
      }
      good += counter.good;
      bad += counter.bad;
      unknown += counter.unknown;
    }
  }

  private class LLAggregator implements Aggregator {
    boolean aggregated = false;
    private final LLCounter[] counters = new LLCounter[grid.size() + grid.rows()];
    private final LLCounter[] left = new LLCounter[grid.size()];
    private final LLCounter[] right = new LLCounter[grid.size()];
    public LLAggregator() {
      for (int i = 0; i < counters.length; i++) {
        counters[i] = new LLCounter();
      }
      for (int i = 0; i < left.length; i++) {
        left[i] = new LLCounter();
        right[i] = new LLCounter();
      }
    }

    @Override
    public void append(int feature, byte bin, double target, double current, double weight) {
      counters[bin2index(feature, bin)].found(current, target, weight);
    }

    private int bin2index(int feature, byte bin) {
      final BFGrid.BFRow row = grid.row(feature);
      return 1 + feature + (bin > 0 ? row.bf(bin - 1).bfIndex : row.bfStart - 1);
    }

    private void aggregate() {
      if (aggregated)
        return;
      for (int f = 0; f < grid.rows(); f++) {
        final BFGrid.BFRow row = grid.row(f);
        for (int b = 0; b < row.size(); b++) {
          for (byte i = 0; i <= row.size(); i++) {
            if (i - 1 < b)
              left[row.bfStart + b].append(counters[bin2index(f, i)]);
            else
              right[row.bfStart + b].append(counters[bin2index(f, i)]);
          }
        }
      }
      aggregated = true;
    }

    public final double left(int bf) {
      aggregate();
      return left[bf].alpha();
    }

    public final double right(int bf) {
      aggregate();
      return right[bf].alpha();
    }

    public int score(final double[] likelihoods) {
      aggregate();
      for (int i = 0; i < likelihoods.length; i++) {
        likelihoods[i] += left[i].score() + right[i].score();
      }
      return ArrayTools.max(likelihoods);
    }
  }
}
