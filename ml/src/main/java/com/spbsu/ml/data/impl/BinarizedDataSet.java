package com.spbsu.ml.data.impl;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;

/**
 * User: solar
 * Date: 05.12.12
 * Time: 21:19
 */
public class BinarizedDataSet {
  private final DataSet base;
  private final BFGrid grid;
  private final byte[][] bins;

  public BinarizedDataSet(final DataSet base, final BFGrid grid) {
    this.base = base;
    this.grid = grid;
    bins = new byte[((VecDataSet) base).xdim()][];
    for (int f = 0; f < bins.length; f++) {
      bins[f] = new byte[base.length()];
    }
    final byte[] binarization = new byte[grid.rows()];
    for (int t = 0; t < base.length(); t++) {
      grid.binarize(((VecDataSet) base).data().row(t), binarization);
      for (int f = 0; f < bins.length; f++) {
        bins[f][t] = binarization[f];
      }
    }
  }


  public DataSet original() {
    return base;
  }

  public BFGrid grid() {
    return grid;
  }

  public byte[] bins(final int findex) {
    return bins[findex];
  }
}
