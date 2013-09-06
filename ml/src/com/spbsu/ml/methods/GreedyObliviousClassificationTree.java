package com.spbsu.ml.methods;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.*;

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
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
    final Vec target = ds instanceof Bootstrap ? ((Bootstrap)ds).original().target() : ds.target();
    Leaf seed = new Leaf(point, target, VecTools.fill(new ArrayVec(point.dim()), 1.));
    Leaf[] leaves1 = new Leaf[]{seed};
    for (int level = 0; level < depth; level++) {
      double[] scores = new double[grid.size()];
      for (Leaf leaf : leaves1) {
        leaf.score(scores);
      }
      final int max = ArrayTools.max(scores);
      if (max < 0)
        throw new RuntimeException("Can not find optimal split!");
      BFGrid.BinaryFeature bestSplit = grid.bf(max);
      final Leaf[] nextLayer = new Leaf[1 << (level + 1)];
      for (int i1 = 0; i1 < leaves1.length; i1++) {
        Leaf leaf = leaves1[i1];
        nextLayer[2 * i1] = leaf;
        nextLayer[2 * i1 + 1] = leaf.split(bestSplit);
      }
      conditions.add(bestSplit);
      leaves1 = nextLayer;
    }
    final Leaf[] leaves = leaves1;

    double[] values = new double[leaves.length];
    double[] weights = new double[leaves.length];
    {
      for (int i = 0; i < weights.length; i++) {
        values[i] = leaves[i].alpha();
        if (Double.isNaN(values[i]))
          System.out.println(leaves[i].alpha());
        weights[i] = leaves[i].indices.length;
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
    private double alpha = Double.NaN;
    private double maclaurinLL0;
    private double maclaurinLL1;
    private double maclaurinLL2;
    private double maclaurinLL3;
    private double maclaurinLL4;

    public double alpha() {
      if (good == 0 || bad == 0)
        return 0;
      if (!Double.isNaN(alpha))
        return alpha;
      final double[] x = new double[3];

      int cnt = MathTools.cubic(x, maclaurinLL4, maclaurinLL3, maclaurinLL2, maclaurinLL1);
      double y = 0.;
      double bestLL = maclaurinLL0;
      for (int i = 0; i < cnt; i++) {
        if (abs(x[i]) < 1 && score(x[i]) > bestLL) {
          y = x[i];
          bestLL = score(y);
        }
      }

      return alpha = log((1. - y) / (1. + y));
    }

    private double score(double x) {
      if (good == 0 || bad == 0)
        return maclaurinLL0;
      return maclaurinLL0 - 2 * maclaurinLL1 * x - maclaurinLL2 * x * x;
    }

    public double score() {
      double alpha = alpha();
      double x = (1 - exp(alpha))/(1 + exp(alpha));
      return score(x) - maclaurinLL0;
    }

    public void found(double current, double target, double weight) {
      final double b = target > 0 ? 1. : -1.;
      final double eab = exp(current*b);
      final double eabPlusOne = eab + 1;
      final double eabMinusOne = eab - 1;
      double denominator = eabPlusOne;
      maclaurinLL0 += weight * log(eab/(1. + eab));
      maclaurinLL1 += weight * b/denominator;
      denominator *= eabPlusOne;
      maclaurinLL2 += weight * 2 * eab/denominator;
      denominator *= eabPlusOne;
      maclaurinLL3 += weight * 2 * b * eab * eabMinusOne /denominator;
      denominator *= eabPlusOne;
      maclaurinLL4 += weight * 2 * eab * eabMinusOne * eabMinusOne / denominator;
      if (b > 0)
        good++;
      else
        bad++;
    }

    public void append(LLCounter counter) {
      maclaurinLL0 += counter.maclaurinLL0;
      maclaurinLL1 += counter.maclaurinLL1;
      maclaurinLL2 += counter.maclaurinLL2;
      maclaurinLL3 += counter.maclaurinLL3;
      maclaurinLL4 += counter.maclaurinLL4;
      good += counter.good;
      bad += counter.bad;
    }

    public void substract(LLCounter counter) {
      maclaurinLL0 -= counter.maclaurinLL0;
      maclaurinLL1 -= counter.maclaurinLL1;
      maclaurinLL2 -= counter.maclaurinLL2;
      maclaurinLL3 -= counter.maclaurinLL3;
      maclaurinLL4 -= counter.maclaurinLL4;
      good -= counter.good;
      bad -= counter.bad;
    }

    public int size() {
      return good + bad;
    }
  }

  public class Leaf implements Aggregator {
    private final Vec point;
    private final Vec target;
    private final Vec weight;
    private int[] indices;
    private final LLCounter[] counters = new LLCounter[grid.size() + grid.rows()];
    private LLCounter total = new LLCounter();

    public Leaf(Vec point, Vec target, Vec weight) {
      this(point, target, weight, null);
    }

    public Leaf(Vec point, Vec target, Vec weight, int[] indices) {
      this.point = point;
      this.target = target;
      this.weight = weight;
      this.indices = indices != null ? indices : ds.original() instanceof Bootstrap ? ((Bootstrap) ds.original()).order() : ArrayTools.sequence(0, ds.original().power());
      for (int i = 0; i < this.indices.length; i++) {
        total.found(point.get(i), target.get(i), weight.get(i));
      }

      for (int i = 0; i < counters.length; i++) {
        counters[i] = new LLCounter();
      }
      ds.aggregate(this, target, point, this.indices);
    }

    private Leaf(int[] points, LLCounter right, Leaf bro) {
      point = bro.point;
      target = bro.target;
      weight = bro.weight;
      this.indices = points;
      total = right;
      for (int i = 0; i < counters.length; i++) {
        counters[i] = new LLCounter();
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

    public int score(final double[] likelihoods) {
      for (int f = 0; f < grid.rows(); f++) {
        final BFGrid.BFRow row = grid.row(f);
        LLCounter left = new LLCounter();
        LLCounter right = new LLCounter();
        right.append(total);

        for (int b = 0; b < row.size(); b++) {
          left.append(counters[bin2index(f, (byte)b)]);
          right.substract(counters[bin2index(f, (byte)b)]);
          likelihoods[row.bfStart + b] = score(left) + score(right);
        }
      }
      return ArrayTools.max(likelihoods);
    }

    private double score(LLCounter counter) {
      return counter.score();// /log(counter.size() + 2);
    }

    /** Splits this leaf into two right side is returned */
    public Leaf split(BFGrid.BinaryFeature feature) {
      LLCounter left = new LLCounter();
      LLCounter right = new LLCounter();
      right.append(total);

      for (int b = 0; b <= feature.binNo; b++) {
        left.append(counters[bin2index(feature.findex, (byte)b)]);
        right.substract(counters[bin2index(feature.findex, (byte)b)]);
      }

      final int[] leftPoints = new int[left.size()];
      final int[] rightPoints = new int[right.size()];
      final Leaf brother = new Leaf(rightPoints, right, this);

      {
        int leftIndex = 0;
        int rightIndex = 0;
        byte[] bins = ds.bins(feature.findex);
        byte splitBin = (byte)feature.binNo;
        for (int i = 0; i < indices.length; i++) {
          final int point = indices[i];
          if (bins[point] > splitBin)
            rightPoints[rightIndex++] = point;
          else
            leftPoints[leftIndex++] = point;
        }
        ds.aggregate(brother, target, point, rightPoints);
      }
      for (int i = 0; i < counters.length; i++) {
        counters[i].substract(brother.counters[i]);
      }
      indices = leftPoints;
      total = left;
      return brother;
    }

    public double alpha() {
      return total.alpha();
    }
  }
}
