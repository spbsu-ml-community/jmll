package com.expleague.ml.dynamicGrid;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.ThreadTools;
import com.expleague.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.expleague.ml.dynamicGrid.interfaces.BinaryFeature;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.dynamicGrid.interfaces.DynamicRow;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.IntFunction;

@SuppressWarnings("unchecked")
public class AggregateDynamic {
  private final BinarizedDynamicDataSet bds;
  private final DynamicGrid grid;
  public final AdditiveStatistics[][] bins;
  private final IntFunction<AdditiveStatistics> factory;
  private int[] points;

  public void updatePoints(final int[] points) {
    this.points = points;
  }

  public AggregateDynamic(final BinarizedDynamicDataSet bds, final IntFunction<AdditiveStatistics> factory, final int[] points) {
    this.points = points;
    this.bds = bds;
    this.grid = bds.grid();
    this.bins = new AdditiveStatistics[grid.rows()][];
    Arrays.fill(bins, new AdditiveStatistics[0]);

    this.factory = factory;
    rebuild(points, ArrayTools.sequence(0, grid.rows()));
  }

  public AdditiveStatistics combinatorForFeature(final BinaryFeature bf) {
    final AdditiveStatistics result = factory.apply(bf.fIndex());
    final DynamicRow row = bf.row();
    final int binNo = bf.binNo();
    final int origFIndex = row.origFIndex();
    for (int b = 0; b <= binNo; b++) {
      result.append(bins[origFIndex][b]);
    }
    return result;
  }

  public AdditiveStatistics total(int findex) {
    final AdditiveStatistics myTotal = factory.apply(findex);
    final DynamicRow row = grid.nonEmptyRow();
    final AdditiveStatistics[] myBins = bins[row.origFIndex()];
    for (AdditiveStatistics myBin : myBins) {
      myTotal.append(myBin);
    }
    return myTotal;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

  public void remove(final AggregateDynamic aggregate) {
    //noinspection UnnecessaryLocalVariable
    final AdditiveStatistics[][] my = bins;
    //noinspection UnnecessaryLocalVariable
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
    for (int f = 0; f < grid.rows(); f++) {
      final T left = (T) factory.apply(f);
      final T right = (T) total(f);
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
      exec.execute(() -> {
        final short[] bin = bds.bins(findex);
        if (!grid.row(findex).empty()) {

          final int length = 4 * (indices.length / 4);
          final AdditiveStatistics[] binsLocal = new AdditiveStatistics[grid.row(findex).size() + 1];

          for (int i = 0; i < binsLocal.length; ++i)
            binsLocal[i] = factory.apply(findex);
//              for (int i : indices) {
//                binsLocal[bin[i]].append(i, 1);
//              }

          //noinspection UnnecessaryLocalVariable
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
          bins[findex] = binsLocal;
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
