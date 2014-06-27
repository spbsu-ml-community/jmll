package com.spbsu.ml.data.impl;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataSet;

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
    bins = new byte[base.xdim()][];
    for (int f = 0; f < bins.length; f++) {
      bins[f] = new byte[base.power()];
    }
    byte[] binarization = new byte[grid.rows()];
    for (int t = 0; t < base.power(); t++) {
      grid.binarize(base.data().row(t), binarization);
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

  public byte[] bins(int findex) {
    return bins[findex];
  }
}
