package com.spbsu.ml;

import com.spbsu.ml.data.impl.BinarizedDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public abstract class BinOptimizedModel extends FuncStub {
  protected final BFGrid grid;

  protected BinOptimizedModel(BFGrid grid) {
    this.grid = grid;
  }

  @Override
  public int xdim() {
    return grid.size();
  }


  protected abstract double value(BinarizedDataSet bds, int index);
}
