package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.softBorders.dataSet.SoftDataSet;
import com.spbsu.ml.data.softBorders.dataSet.SoftGrid;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */
@SuppressWarnings("unchecked")
public class SoftAggregate {
  private final SoftDataSet softDataSet;
  private final SoftGrid grid;
  private final AdditiveStatistics[] bins;
  private final int[] starts;
  private final Factory<AdditiveStatistics> factory;

  public SoftAggregate(final SoftDataSet dataSet,
                       final Factory<AdditiveStatistics> factory
  ) {
    this.softDataSet = dataSet;
    this.grid = dataSet.grid();
    this.starts = new int[grid.rowsCount()];
    int binsSize = 0;
    for (int i = 0; i < grid.rowsCount(); ++i) {
      if (grid.binFeatureCount() > 0) {
        starts[i] = binsSize;
        binsSize += grid.row(i).binFeatureCount() + 1;
      }
    }
    this.bins = new AdditiveStatistics[binsSize];
    for (int i = 0; i < bins.length; i++) {
      bins[i] = factory.create();
    }
    this.factory = factory;
  }

  public SoftAggregate(final SoftDataSet dataSet,
                       final Factory<AdditiveStatistics> factory,
                       final int[] points) {
    this(dataSet, factory);
    build(points);
  }


  public AdditiveStatistics total() {
    final AdditiveStatistics myTotal = factory.create();
    final SoftGrid.SoftRow row = grid.nonEmptyRow();
    final int offset = starts[row.featureIdx()];
    for (int b = 0; b <= row.binFeatureCount(); b++) {
      myTotal.append(bins[offset + b]);
    }
    return myTotal;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Soft aggregator thread", -1);

  public void remove(final SoftAggregate aggregate) {
    final AdditiveStatistics[] my = bins;
    final AdditiveStatistics[] other = aggregate.bins;
    for (int i = 0; i < bins.length; i++) {
      my[i].remove(other[i]);
    }
  }

  public void append(final SoftAggregate aggregate) {
    final AdditiveStatistics[] my = bins;
    final AdditiveStatistics[] other = aggregate.bins;
    for (int i = 0; i < bins.length; i++) {
      my[i].append(other[i]);
    }
  }

  public interface SplitVisitor<T> {
    void accept(SoftGrid.SoftRow.BinFeature bf, T left, T right);
  }

  public AdditiveStatistics combinatorForFeature(final int bf) {
    final AdditiveStatistics result = factory.create();
    final SoftGrid.SoftRow row = grid.bf(bf).row();
    final int binNo = grid.bf(bf).binIdx;
    final int offset = starts[row.featureIdx()];
    for (int b = 0; b <= binNo; b++) {
      result.append(bins[offset + b]);
    }
    return result;
  }

  //take bf and next (length-1) binary features as one
  public interface IntervalVisitor<T> {
    void accept(SoftGrid.SoftRow.BinFeature row, int startBin, int endBin, T inside, T outside);
  }


  public <T extends AdditiveStatistics> void visit(final SplitVisitor<T> visitor) {
    final T total = (T) total();
    for (int f = 0; f < grid.rowsCount(); f++) {
      final T left = (T) factory.create();
      final T right = (T) factory.create().append(total);
      final SoftGrid.SoftRow row = grid.row(f);
      final int offset = starts[row.featureIdx()];
      for (int b = 0; b < row.binFeatureCount(); b++) {
        left.append(bins[offset + b]);
        right.remove(bins[offset + b]);
        visitor.accept(row.binFeature(b), left, right);
      }
    }
  }

  private void build(final int[] indices) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rowsCount());
    for (int findex = 0; findex < grid.rowsCount(); findex++) {
      final SoftGrid.SoftRow row = grid.row(findex);
      exec.execute(() -> {
        final int offset = starts[row.featureIdx()];
        if (row.binFeatureCount() > 0) {
          for (int i : indices) {
            Vec binsDistr = softDataSet.binDistribution(row.featureIdx(), i);
            for (int bin = 0; bin < binsDistr.dim(); ++bin) {
              final double weight = binsDistr.get(bin);
              if (weight > 1e-9) {
                bins[offset + bin].append(i, weight);
              }
            }
//            bins[offset + softDataSet.bin(row.featureIdx(), i)].append(i, 1);
          }
        }
        latch.countDown();
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
  }

}
