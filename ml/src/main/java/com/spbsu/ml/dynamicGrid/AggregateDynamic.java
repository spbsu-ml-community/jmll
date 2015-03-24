package com.spbsu.ml.dynamicGrid;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicRow;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

@SuppressWarnings("unchecked")
public class AggregateDynamic {
  private final BinarizedDynamicDataSet bds;
  private final DynamicGrid grid;
  public final AdditiveStatistics[][] bins;
  private final Factory<AdditiveStatistics> factory;
  private int[] points;

  public void updatePoints(final int[] points) {
    this.points = points;
  }

  public AggregateDynamic(final BinarizedDynamicDataSet bds, final Factory<AdditiveStatistics> factory, final int[] points) {
    this.points = points;
    this.bds = bds;
    this.grid = bds.grid();
    this.bins = new AdditiveStatistics[grid.rows()][];
    for (int feat = 0; feat < bins.length; feat++) {
      bins[feat] = new AdditiveStatistics[0];
    }

    this.factory = factory;
    rebuild(points, ArrayTools.sequence(0, grid.rows()));
  }

  public AdditiveStatistics combinatorForFeature(final BinaryFeature bf) {
    final AdditiveStatistics result = factory.create();
    final DynamicRow row = bf.row();
    final int binNo = bf.binNo();
    final int origFIndex = row.origFIndex();
    for (int b = 0; b <= binNo; b++) {
      result.append(bins[origFIndex][b]);
    }
    return result;
  }

  public AdditiveStatistics total() {
    final AdditiveStatistics myTotal = factory.create();
    final DynamicRow row = grid.nonEmptyRow();
    final AdditiveStatistics[] myBins = bins[row.origFIndex()];
    for (int bin = 0; bin < myBins.length; ++bin) {
      myTotal.append(myBins[bin]);
    }
    return myTotal;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

  public void remove(final AggregateDynamic aggregate) {
    final AdditiveStatistics[][] my = bins;
    final AdditiveStatistics[][] other = aggregate.bins;
    for (int i = 0; i < bins.length; i++) {
      for (int j = 0; j < bins[i].length; ++j) {
        my[i][j].remove(other[i][j]);
      }
    }
  }

  public interface SplitVisitor<T> {
    void accept(BinaryFeature bf, T left, T right);
  }


//  private static int sequentialLimit = 4;
//
//  public <T extends AdditiveStatistics> void visit(final SplitVisitor<T> visitor) {
//    final CountDownLatch latch = new CountDownLatch(grid.rows());
//    final T total = (T) total();
//    for (int f = 0; f < grid.rows(); f++) {
//      final DynamicRow row = grid.row(f);
//      if (row.size() < sequentialLimit) {
//        final T left = (T) factory.create();
//        final T right = (T) factory.create().append(total);
//        final AdditiveStatistics[] rowBins = bins[row.origFIndex()];
//        for (int b = 0; b < row.size(); b++) {
//          left.append(rowBins[b]);
//          right.remove(rowBins[b]);
//          visitor.accept(row.bf(b), left, right);
//        }
//        latch.countDown();
//      } else {
//        exec.execute(new Runnable() {
//          @Override
//          public void run() {
//            final T left = (T) factory.create();
//            final T right = (T) factory.create().append(total);
//            final AdditiveStatistics[] rowBins = bins[row.origFIndex()];
//            for (int b = 0; b < row.size(); b++) {
//              left.append(rowBins[b]);
//              right.remove(rowBins[b]);
//              visitor.accept(row.bf(b), left, right);
//            }
//            latch.countDown();
//          }
//        });
//      }
//    }
//    try {
//      latch.await();
//    } catch (InterruptedException e) {
//      // skip
//    }
//  }

  public <T extends AdditiveStatistics> void visit(final SplitVisitor<T> visitor) {
    final T total = (T) total();
    for (int f = 0; f < grid.rows(); f++) {
      final T left = (T) factory.create();
      final T right = (T) factory.create().append(total);
      final DynamicRow row = grid.row(f);
      final AdditiveStatistics[] rowBins = bins[row.origFIndex()];
      for (int b = 0; b < row.size(); b++) {
        left.append(rowBins[b]);
        right.remove(rowBins[b]);
        visitor.accept(row.bf(b), left, right);
      }
    }
  }

  public void rebuild(final int... features) {
    rebuild(this.points, features);
  }

  private void rebuild(final int[] indices, final int... features) {
    final CountDownLatch latch = new CountDownLatch(features.length);
    for (final int findex : features) {
      final int finalFIndex = findex;
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final short[] bin = bds.bins(finalFIndex);
          if (!grid.row(finalFIndex).empty()) {

            final int length = 4 * (indices.length / 4);
            final AdditiveStatistics[] binsLocal = new AdditiveStatistics[grid.row(finalFIndex).size() + 1];

            for (int i = 0; i < binsLocal.length; ++i)
              binsLocal[i] = factory.create();
//              for (int i : indices) {
//                binsLocal[bin[i]].append(i, 1);
//              }
            final int[] indicesLocal = indices;
            for (int i = 0; i < length; i += 4) {
              final int idx1 = indicesLocal[i];
              final int idx2 = indicesLocal[i + 1];
              final int idx3 = indicesLocal[i + 2];
              final int idx4 = indicesLocal[i + 3];
              final AdditiveStatistics bin1 = binsLocal[bin[idx1]];
              final AdditiveStatistics bin2 = binsLocal[bin[idx2]];
              final AdditiveStatistics bin3 = binsLocal[bin[idx3]];
              final AdditiveStatistics bin4 = binsLocal[bin[idx4]];
              bin1.append(idx1, 1);
              bin2.append(idx2, 1);
              bin3.append(idx3, 1);
              bin4.append(idx4, 1);
            }
            for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
              binsLocal[bin[indicesLocal[i]]].append(indicesLocal[i], 1);
            }
            bins[finalFIndex] = binsLocal;
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
}
