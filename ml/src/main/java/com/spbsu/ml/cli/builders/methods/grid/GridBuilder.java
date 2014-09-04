package com.spbsu.ml.cli.builders.methods.grid;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.data.set.VecDataSet;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GridBuilder implements Factory<BFGrid> {
  BFGrid cooked;
  int binsCount = 32;
  VecDataSet dataSet;

  public void setGrid(final BFGrid cooked) {
    this.cooked = cooked;
  }

  public void setBinsCount(final int binsCount) {
    this.binsCount = binsCount;
  }

  public void setDataSet(final VecDataSet dataSet) {
    this.dataSet = dataSet;
  }

  @Override
  public BFGrid create() {
    if (cooked == null) {
      cooked = GridTools.medianGrid(dataSet, binsCount);
    }
    return cooked;
  }
}
