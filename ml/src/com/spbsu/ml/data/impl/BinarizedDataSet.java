package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.Histogram;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 05.12.12
 * Time: 21:19
 */
public class BinarizedDataSet {
  private final DataSet base;
  private final BFGrid grid;
  private final byte[][] bins;

  public BinarizedDataSet(DataSet base, BFGrid grid) {
    this.base = base;
    this.grid = grid;
    exec = ThreadTools.createExecutor("BDS Aggregator thread", grid.size());
    bins = new byte[base.xdim()][];
    for (int f = 0; f < bins.length; f++) {
      bins[f] = new byte[base.power()];
    }
    byte[] binarization = new byte[grid.size()];
    for (int t = 0; t < base.power(); t++) {
      grid.binarize(base.data().row(t), binarization);
      for (int f = 0; f < bins.length; f++) {
        bins[f][t] = binarization[f];
      }
    }
  }

  public static final int POOL_SIZE = Runtime.getRuntime().availableProcessors();
  private final ThreadPoolExecutor exec;

  public DataSet original() {
    return base;
  }

  public BFGrid grid() {
    return grid;
  }

  public Histogram buildHistogram(final Vec target, final Vec point, final int[] indices) {
    final Histogram result = new Histogram(grid);
    aggregate(result, target, point, indices);
    return result;
  }

  public void aggregate(final Aggregator aggregator, final Vec target, final Vec point, final int[] indices) {
    final CountDownLatch latch = new CountDownLatch(grid.rows());
    for (int findex = 0; findex < grid.rows(); findex++) {
      final int finalFIndex = findex;
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final byte[] bin = bins[finalFIndex];
          if (!grid.row(finalFIndex).empty()) {
            for (int t = 0; t < indices.length; t++) {
              final int pindex = indices[t];
              aggregator.append(finalFIndex, bin[pindex], target.get(pindex), point.get(pindex), 1.);
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

  public byte[] bins(int findex) {
    return bins[findex];
  }
}
