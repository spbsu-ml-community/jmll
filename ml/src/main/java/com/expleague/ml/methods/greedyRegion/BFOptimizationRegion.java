package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.math.AnalyticFunc;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.impl.BinaryFeatureImpl;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.WeightedLoss;
import gnu.trove.list.array.TIntArrayList;


/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:16
 */
@SuppressWarnings("unchecked")
public class BFOptimizationRegion {
  protected final BinarizedDataSet bds;
  protected int[] pointsInside;
  protected final AdditiveLoss<?> oracle;
  protected Aggregate aggregate;

  public BFOptimizationRegion(final BinarizedDataSet bds,
                              final AdditiveLoss<?> oracle,
                              final int[] points) {
    this.bds = bds;
    this.pointsInside = points;
    this.oracle = oracle;
    this.aggregate = new Aggregate(bds, oracle.statsFactory());
    this.aggregate.append(points);
  }

  public void split(final BFGrid.Feature feature, final boolean mask) {
    final byte[] bins = bds.bins(feature.findex());

    final TIntArrayList newInside = new TIntArrayList();
    final TIntArrayList newOutside = new TIntArrayList();


    for (int index : pointsInside) {
      if ((bins[index] > feature.bin()) != mask) {
        newOutside.add(index);
      } else {
        newInside.add(index);
      }
    }
    pointsInside = newInside.toArray();
    if (newInside.size() < newOutside.size()) {
      aggregate = new Aggregate(bds, oracle.statsFactory());
      aggregate.append(pointsInside);
    }
    else aggregate.remove(newOutside.toArray());
  }

  int size() {
    return pointsInside.length;
  }

  public void visitAllSplits(final Aggregate.SplitVisitor<? extends AdditiveStatistics> visitor) {
    aggregate.visit(visitor);
  }

  public <T extends AdditiveStatistics> void visitSplit(final BinaryFeatureImpl bf, final Aggregate.SplitVisitor<T> visitor) {
    final T left = (T) aggregate.combinatorForFeature(bf.bfIndex);
    final T right = (T) oracle.statsFactory().apply(bf.findex).append(aggregate.total(-1)).remove(left);
    visitor.accept(bf, left, right);
  }

  public AdditiveStatistics total() {
    return aggregate.total(-1);
  }

  public static class PermutationWeightedFunc extends AnalyticFunc.Stub {
    private final int c;
    private final Aggregate aggregate;
    private final WeightedLoss<? extends L2> loss;
    private final int[] order;

    public PermutationWeightedFunc(int c, int[] order, Aggregate aggregate, WeightedLoss<? extends L2> loss) {
      this.c = c;
      this.order = order;
      this.aggregate = aggregate;
      this.loss = loss;
    }

    @Override
    public double value(double x) {
      double[] params = new double[]{0, 0, 0};
      aggregate.visitND(c, order.length, x, (k, N_k, D_k, P_k, S_k) -> {
        final int index = order[k];
        final double y_k = loss.base().target().get(index);
        final double w_k = loss.weight(index) * N_k / D_k;

        params[0] += w_k * y_k * y_k;
        params[1] += w_k * y_k;
        params[2] += w_k;
      });
      double sum2 = params[0];
      double sum = params[1];
      double weights = params[2];
      return sum2 - sum * sum / weights;
    }

    @Override
    public double gradient(double x) {
      final double[] params = new double[]{0};
      final WeightedLoss.Stat stat = (WeightedLoss.Stat) aggregate.total(-1);
      final L2.Stat l2Stat = (L2.Stat)stat.inside;
      aggregate.visitND(c, order.length, x, (k, N_k, D_k, P_k, S_k) -> {
        final int index = order[k];
        final double y_k = loss.base().target().get(index);
        final double w_k = loss.weight(index) * N_k / D_k;

        final double dLdw = y_k * y_k - 2 * (y_k * l2Stat.sum * l2Stat.weight - l2Stat.sum * l2Stat.sum) / l2Stat.weight / l2Stat.weight / l2Stat.weight;
        final double dwdl = (S_k * D_k - P_k * N_k) / N_k / N_k;
        params[0] += w_k * dLdw * dwdl;
      });
      return params[0];
    }
  }
}
