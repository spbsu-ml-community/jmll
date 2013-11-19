package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveGator;
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
  int transferRowSize;
  private final BFGrid grid;
  private final AdditiveGator[] bins;
  private final Factory<AdditiveGator> factory;
  private AdditiveGator total;

  public Aggregate(BinarizedDataSet bds, Factory<AdditiveGator> factory, int[] points) {
    this.bds = bds;
    this.grid = bds.grid();
    int maxRow = 0;
    for (int i = 0; i < grid.rows(); i++) {
      maxRow = Math.max(maxRow, grid.row(i).size()) + 1;
    }
    transferRowSize = maxRow;
    this.bins = new AdditiveGator[transferRowSize * grid.rows()];
    for (int i = 0; i < bins.length; i++) {
      bins[i] = factory.create();

    }
    this.factory = factory;
    build(points);
  }

  public AdditiveGator combinatorForFeature(int bf) {
    final AdditiveGator result = factory.create();

    final BFGrid.BFRow row = grid.bf(bf).row();
    final int binNo = grid.bf(bf).binNo;
    final int offset = row.origFIndex * transferRowSize;
    for (int b = 0; b <= binNo; b++) {
      result.append(bins[offset + b]);
    }
    return result;
  }

  public synchronized AdditiveGator total() {
    if (total == null) { // calculating total by non empty row
      total = factory.create();
      final BFGrid.BFRow row = grid.nonEmptyRow();
      for (int b = 0; b <= row.size(); b++) {
        total.append(bins[row.origFIndex * transferRowSize + b]);
      }
    }

    return total;
  }

  public void remove(Aggregate aggregate) {
    for (int i = 0; i < bins.length; i++) {
      bins[i].remove(aggregate.bins[i]);
    }
    total.remove(aggregate.total());
  }

  public interface SplitVisitor<T> {
    void accept(BFGrid.BinaryFeature bf, T left, T right);
  }

  public <T extends AdditiveGator> void visit(SplitVisitor<T> visitor) {
    final T total = (T)total();

    for (int f = 0; f < grid.rows(); f++) {
      final T left = (T)factory.create();
      final T right = (T)factory.create().append(total);
      final BFGrid.BFRow row = grid.row(f);
      for (int b = 0; b < row.size(); b++) {
        left.append(bins[row.origFIndex * transferRowSize + b]);
        right.remove(bins[row.origFIndex * transferRowSize + b]);
        visitor.accept(row.bf(b), left, right);
      }
    }
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Aggregator thread", -1);

  private void build(final int[] indices) {
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final int finalFIndex = findex;
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final byte[] bin = bds.bins(finalFIndex);
          final int offset = finalFIndex * transferRowSize;
          if (!grid.row(finalFIndex).empty()) {
            for (int i : indices) {
              bins[offset + bin[i]].append(i);
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
}
