package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */
@SuppressWarnings("unchecked")
public class Aggregate {
  private final BinarizedDataSet bds;
  private final BFGrid grid;
  private final AdditiveStatistics[] bins;
  private final int[] starts;
  private final Factory<AdditiveStatistics> factory;

  public Aggregate(final BinarizedDataSet bds, final Factory<AdditiveStatistics> factory, final int[] points) {
    this(bds, factory);
    build(points);
  }

  public Aggregate(final BinarizedDataSet bds, final Factory<AdditiveStatistics> factory, final int[] points, final double[] weights) {
    this(bds, factory);
    build(points, weights);
  }

  public Aggregate(final BinarizedDataSet bds, final Factory<AdditiveStatistics> factory) {
    this.bds = bds;
    this.grid = bds.grid();
    this.starts = new int[grid.rows()];
    int binsSize = 0;
    for (int i = 0; i < grid.rows(); ++i) {
      starts[i] = binsSize;
      binsSize += grid.row(i).size() + 1;
    }
    this.bins = new AdditiveStatistics[binsSize];
    for (int i = 0; i < bins.length; i++) {
      bins[i] = factory.create();
    }
    this.factory = factory;
  }

  public AdditiveStatistics combinatorForFeature(final int bf) {
    final AdditiveStatistics result = factory.create();
    final BFGrid.BFRow row = grid.bf(bf).row();
    final int binNo = grid.bf(bf).binNo;
    final int offset = starts[row.origFIndex];
    for (int b = 0; b <= binNo; b++) {
      result.append(bins[offset + b]);
    }
    return result;
  }

  public AdditiveStatistics total() {
    final AdditiveStatistics myTotal = factory.create();
    final BFGrid.BFRow row = grid.nonEmptyRow();
    final int offset = starts[row.origFIndex];
    final AdditiveStatistics[] myBins = bins;
    for (int b = 0; b <= row.size(); b++) {
      myTotal.append(myBins[offset + b]);
    }
    return myTotal;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

  public void remove(final Aggregate aggregate) {
    final AdditiveStatistics[] my = bins;
    final AdditiveStatistics[] other = aggregate.bins;
    for (int i = 0; i < bins.length; i++) {
      my[i].remove(other[i]);
    }
  }

  public void append(final Aggregate aggregate) {
    final AdditiveStatistics[] my = bins;
    final AdditiveStatistics[] other = aggregate.bins;
    for (int i = 0; i < bins.length; i++) {
      my[i].append(other[i]);
    }
  }

  public interface SplitVisitor<T> {
    void accept(BFGrid.BinaryFeature bf, T left, T right);
  }


  //take bf and next (length-1) binary features as one
  public interface IntervalVisitor<T> {
    void accept(BFGrid.BFRow row, int startBin, int endBin, T inside, T outside);
  }


  public <T extends AdditiveStatistics> void visit(final SplitVisitor<T> visitor) {
    final T total = (T) total();
    for (int f = 0; f < grid.rows(); f++) {
      final T left = (T) factory.create();
      final T right = (T) factory.create().append(total);
      final BFGrid.BFRow row = grid.row(f);
      final int offset = starts[row.origFIndex];
      for (int b = 0; b < row.size(); b++) {
        left.append(bins[offset + b]);
        right.remove(bins[offset + b]);
        visitor.accept(row.bf(b), left, right);
      }
    }
  }

  public <T extends AdditiveStatistics> void visit(final IntervalVisitor<T> visitor) {
    final T total = (T) total();
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int f = 0; f < grid.rows(); f++) {
      final BFGrid.BFRow row = grid.row(f);
      final int offset = starts[row.origFIndex];
      exec.submit(new Runnable() {
        @Override
        public void run() {
          if (!row.empty())
          for (int startBin = 0; startBin <= row.size(); ++startBin) {
            final T inside = (T) factory.create();
            final T outside = (T) factory.create().append(total);
            for (int endBin = startBin; endBin <= row.size(); ++endBin) {
              inside.append(bins[offset + endBin]);
              outside.remove(bins[offset + endBin]);
              visitor.accept(row, startBin, endBin, inside, outside);
            }
            ++startBin;
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      //
    }
  }


  private void build(final int[] indices) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final BFGrid.BFRow row = grid.row(findex);
      final byte[] bin = bds.bins(findex);
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final int offset = starts[row.origFIndex];
          if (!row.empty()) {
//            for (int i : indices) {
//              bins[offset + bin[i]].append(i, 1);
//            }
            final int length = 4 * (indices.length / 4);
            final AdditiveStatistics[] binsLocal = bins;
            @SuppressWarnings("UnnecessaryLocalVariable")
            final int[] indicesLocal = indices;
            @SuppressWarnings("UnnecessaryLocalVariable")
            final byte[] binLocal = bin;
            for (int i = 0; i < length; i += 4) {
              final int idx1 = indicesLocal[i];
              final int idx2 = indicesLocal[i + 1];
              final int idx3 = indicesLocal[i + 2];
              final int idx4 = indicesLocal[i + 3];
              final AdditiveStatistics bin1 = binsLocal[offset + binLocal[idx1]];
              final AdditiveStatistics bin2 = binsLocal[offset + binLocal[idx2]];
              final AdditiveStatistics bin3 = binsLocal[offset + binLocal[idx3]];
              final AdditiveStatistics bin4 = binsLocal[offset + binLocal[idx4]];
              bin1.append(idx1, 1);
              bin2.append(idx2, 1);
              bin3.append(idx3, 1);
              bin4.append(idx4, 1);
            }
            for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
              binsLocal[offset + bin[indicesLocal[i]]].append(indicesLocal[i], 1);
            }
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
  }


  public void append(final int[] indices) {
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final int finalFIndex = findex;
      final BFGrid.BFRow row = grid.row(findex);
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final byte[] bin = bds.bins(finalFIndex);
          final int offset = starts[row.origFIndex];
          if (!row.empty()) {
//            for (int i : indices) {
//              bins[offset + bin[i]].append(i, 1);
//            }
            final int length = 4 * (indices.length / 4);
            final AdditiveStatistics[] binsLocal = bins;
            final int[] indicesLocal = indices;
            for (int i = 0; i < length; i += 4) {
              final int idx1 = indicesLocal[i];
              final int idx2 = indicesLocal[i + 1];
              final int idx3 = indicesLocal[i + 2];
              final int idx4 = indicesLocal[i + 3];
              final AdditiveStatistics bin1 = binsLocal[offset + bin[idx1]];
              final AdditiveStatistics bin2 = binsLocal[offset + bin[idx2]];
              final AdditiveStatistics bin3 = binsLocal[offset + bin[idx3]];
              final AdditiveStatistics bin4 = binsLocal[offset + bin[idx4]];
              bin1.append(idx1, 1);
              bin2.append(idx2, 1);
              bin3.append(idx3, 1);
              bin4.append(idx4, 1);
            }
            for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
              binsLocal[offset + bin[indicesLocal[i]]].append(indicesLocal[i], 1);
            }
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
  }

  private void build(final int[] indices, final double[] weights) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final BFGrid.BFRow row = grid.row(findex);
      final byte[] bin = bds.bins(findex);
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final int offset = starts[row.origFIndex];
          if (!row.empty()) {
//            for (int i : indices) {
//              bins[offset + bin[i]].append(i, 1);
//            }
            final int length = 4 * (indices.length / 4);
            final AdditiveStatistics[] binsLocal = bins;
            @SuppressWarnings("UnnecessaryLocalVariable")
            final int[] indicesLocal = indices;
            @SuppressWarnings("UnnecessaryLocalVariable")
            final byte[] binLocal = bin;
            for (int i = 0; i < length; i += 4) {
              final int idx1 = indicesLocal[i];
              final int idx2 = indicesLocal[i + 1];
              final int idx3 = indicesLocal[i + 2];
              final int idx4 = indicesLocal[i + 3];
              final AdditiveStatistics bin1 = binsLocal[offset + binLocal[idx1]];
              final AdditiveStatistics bin2 = binsLocal[offset + binLocal[idx2]];
              final AdditiveStatistics bin3 = binsLocal[offset + binLocal[idx3]];
              final AdditiveStatistics bin4 = binsLocal[offset + binLocal[idx4]];
              bin1.append(idx1, weights[i]);
              bin2.append(idx2, weights[i + 1]);
              bin3.append(idx3, weights[i + 2]);
              bin4.append(idx4, weights[i + 3]);
            }
            for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
              binsLocal[offset + bin[indicesLocal[i]]].append(indicesLocal[i], weights[i]);
            }
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
    //need for cherry pick
//    AdditiveStatistics total = total();
//    for (int findex=0; findex < grid.rows();++findex) {
//      final BFGrid.BFRow row = grid.row(findex);
//      if (row.empty()) {
//        final int offset = starts[row.origFIndex];
//        bins[offset] = factory.create().append(total);
//      }
//    }
  }
}
