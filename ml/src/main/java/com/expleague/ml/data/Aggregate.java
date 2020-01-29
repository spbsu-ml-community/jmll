package com.expleague.ml.data;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.util.ThreadTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.impl.BinarizedDataSet;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

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
  private final IntFunction<? extends AdditiveStatistics> factory;
  private final BFGrid.Row nonEmptyRow;

  public Aggregate(final BinarizedDataSet bds, final IntFunction<? extends AdditiveStatistics> factory) {
    this.bds = bds;
    this.grid = bds.grid();
    this.starts = new int[grid.rows()];
    int binsSize = 0;
    BFGrid.Row neRow = null;
    for (int i = 0; i < grid.rows(); ++i) {
      starts[i] = binsSize;
      binsSize += grid.row(i).size() + 1;
      if (grid.row(i).size() != 0)
        neRow = grid.row(i);
    }
    if (neRow == null)
      throw new IllegalArgumentException("All rows of the given grid are empty!");
    nonEmptyRow = neRow;
    this.bins = new AdditiveStatistics[binsSize];
    for (int i = 0, idx = 0; i < grid.rows(); ++i) {
      for (int b = 0; b < grid.row(i).size() + 1; b++, idx++) {
        bins[idx] = factory.apply(i);
      }
    }
    this.factory = factory;
  }

  public AdditiveStatistics combinatorForFeature(final int bf) {
    final BFGrid.Row row = grid.bf(bf).row();
    final AdditiveStatistics result = factory.apply(row.findex());
    final int binNo = grid.bf(bf).bin();
    final int offset = starts[row.findex()];
    for (int b = 0; b <= binNo; b++) {
      result.append(bins[offset + b]);
    }
    return result;
  }

  public AdditiveStatistics total(int findex) {
    final AdditiveStatistics myTotal = factory.apply(findex);
    final int offset = starts[findex];
    int fSize = grid.row(findex).size();
    for (int b = 0; b <= fSize; b++) {
      myTotal.append(bins[offset + b]);
    }
    return myTotal;
  }

  public AdditiveStatistics bin(int findex, int bin) {
    return bins[starts[findex] + bin];
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

  public void remove(final Aggregate aggregate) {
    for (int i = 0; i < bins.length; i++) {
      bins[i].remove(aggregate.bins[i]);
    }
  }

  public void append(final Aggregate aggregate) {
    for (int i = 0; i < bins.length; i++) {
      bins[i].append(aggregate.bins[i]);
    }
  }

  public interface SplitVisitor<T> {
    void accept(BFGrid.Feature bf, T left, T right);
  }


  //take bf and next (length-1) binary features as one
  public interface IntervalVisitor<T> {
    void accept(BFGrid.Row row, int startBin, int endBin, T inside, T outside);
  }

  public <T extends AdditiveStatistics> void visit(final SplitVisitor<T> visitor) {
    IntStream.range(0, grid.rows()).parallel().forEach(f -> {
      final T total = (T) total(f);
      final T left = (T) factory.apply(f);
      final T right = (T) factory.apply(f).append(total);
      final BFGrid.Row row = grid.row(f);
      final int offset = starts[row.findex()];

      if (!row.ordered()) {
        for (int b = 0; b < row.size(); b++) {
          final T inside = (T) factory.apply(f).append(bins[offset + b]);
          final T outside = (T) factory.apply(f).append(total).remove(inside);
          visitor.accept(row.bf(b), inside, outside);
        }
      } else {
        for (int b = 0; b < row.size(); b++) {
          left.append(bins[offset + b]);
          right.remove(bins[offset + b]);
          visitor.accept(row.bf(b), left, right);
        }
      }
    });
  }

  public <T extends AdditiveStatistics> void visit(final IntervalVisitor<T> visitor) {
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int f = 0; f < grid.rows(); f++) {
      final BFGrid.Row row = grid.row(f);
      final T total = (T) total(f);
      final int offset = starts[row.findex()];
      exec.submit(() -> {
        if (!row.empty())
        for (int startBin = 0; startBin <= row.size(); ++startBin) {
          final T inside = (T) factory.apply(row.findex());
          final T outside = (T) factory.apply(row.findex()).append(total);
          for (int endBin = startBin; endBin <= row.size(); ++endBin) {
            inside.append(bins[offset + endBin]);
            outside.remove(bins[offset + endBin]);
            visitor.accept(row, startBin, endBin, inside, outside);
          }
          ++startBin;
        }
        latch.countDown();
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      //
    }
  }

  public void append(final int[] indices) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final int finalFIndex = findex;
      final BFGrid.Row row = grid.row(findex);
      final int offset = starts[row.findex()];
      exec.execute(() -> {
        final byte[] bin = bds.bins(finalFIndex);
        if (!row.empty()) {
//            for (int i : indices) {
//              bins[offset + bin[i]].append(i, 1);
//            }
          final int length = 4 * (indices.length / 4);
          final AdditiveStatistics[] binsLocal = bins;
          //noinspection UnnecessaryLocalVariable
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
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
  }

  public void append(final int[] indices, final double[] weights) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final BFGrid.Row row = grid.row(findex);
      final byte[] bin = bds.bins(findex);
      exec.execute(() -> {
        final int offset = starts[row.findex()];
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
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      // skip
    }
  }

  public void remove(final int[] indices) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final BFGrid.Row row = grid.row(findex);
      final byte[] bin = bds.bins(findex);
      exec.execute(() -> {
        final int offset = starts[row.findex()];
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
            bin1.remove(idx1, 1);
            bin2.remove(idx2, 1);
            bin3.remove(idx3, 1);
            bin4.remove(idx4, 1);
          }
          for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
            binsLocal[offset + bin[indicesLocal[i]]].remove(indicesLocal[i], 1);
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

  public void remove(final int[] indices, final double[] weights) {
    if (indices.length == 0)
      return;
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final BFGrid.Row row = grid.row(findex);
      final byte[] bin = bds.bins(findex);
      exec.execute(() -> {
        final int offset = starts[row.findex()];
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
            bin1.remove(idx1, weights[i]);
            bin2.remove(idx2, weights[i + 1]);
            bin3.remove(idx3, weights[i + 2]);
            bin4.remove(idx4, weights[i + 3]);
          }
          for (int i = 4 * (indicesLocal.length / 4); i < indicesLocal.length; i++) {
            binsLocal[offset + bin[indicesLocal[i]]].remove(indicesLocal[i], weights[i]);
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

  public interface NDConsumer {
    void visit(int k, double N_k, double D_k, double S_k, double P_k);
  }

  public void visitND(int c, int n, double lambda, NDConsumer consumer) {
    double N_k;
    double D_k = 0;
    double S_k = 0;
    double P_k;

    for (int i = 0; i < c; i++) {
      D_k += Math.exp(-lambda * i);
      S_k += i * Math.exp(-lambda * i);
    }
    N_k = -D_k;
    P_k = -S_k;
    for (int i = c; i < n; i++) {
      D_k += Math.exp(-lambda * i);
      S_k += i * Math.exp(-lambda * i);
    }
    N_k += D_k;
    P_k += S_k;
    consumer.visit(0, N_k, D_k, P_k, S_k);
    for (int k = 1; k < n; k++) {
      final double lastComponent = Math.exp(-lambda * (n - k - 1));
      N_k += Math.exp(-lambda * Math.abs(c - k)) - lastComponent;
      D_k += Math.exp(-lambda * k) - lastComponent;
      final double lastSComponent = (n - k - 1) * Math.exp(-lambda * (n - k - 1));
      P_k += Math.abs(c - k) * Math.exp(-lambda * Math.abs(c - k)) - lastSComponent;
      S_k += k * Math.exp(-lambda * k) - lastSComponent;
      consumer.visit(k, N_k, D_k, P_k, S_k);
    }
  }
}
