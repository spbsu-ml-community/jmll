package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.Histogram;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * User: solar
 * Date: 05.12.12
 * Time: 21:19
 */
public class BinarizedDataSet {
  DataSet base;
  BFGrid grid;
  byte[][] bins;

  public BinarizedDataSet(DataSet base, BFGrid grid) {
    this.base = base;
    this.grid = grid;
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
  private final ThreadPoolExecutor exec = new ThreadPoolExecutor(0, POOL_SIZE, 1, TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(10000));


  public Histogram buildHistogram(final Vec target, final Vec point, final int[] indices) {
    final Histogram result = new Histogram(grid);
    aggregate(result, target, point, indices);
    return result;
  }

  public void aggregate(final Aggregator aggregator, final Vec target, final Vec point, final int[] indices) {
    final CountDownLatch latch = new CountDownLatch(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; i++) {
      final int windex = i;
      exec.execute(new Runnable() {
        @Override
        public void run() {
          final double fsize = Math.ceil(grid.rows() / (double) POOL_SIZE);
          for (int f = 0; f < fsize; f++) {
            int findex = f * POOL_SIZE + windex;
            if (findex >= grid.rows())
              break;
            final byte[] bin = bins[findex];
            if (grid.row(findex).empty())
              continue;
            for (int t = 0; t < indices.length; t++) {
              final int pindex = indices[t];
              aggregator.append(findex, bin[pindex], target.get(pindex), point.get(pindex), 1.);
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
