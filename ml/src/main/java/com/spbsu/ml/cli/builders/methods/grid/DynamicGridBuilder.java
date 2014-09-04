package com.spbsu.ml.cli.builders.methods.grid;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.dynamicGrid.impl.BFDynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class DynamicGridBuilder implements Factory<DynamicGrid> {
  DynamicGrid cooked;
  int binsCount = 32;
  VecDataSet dataSet;

  public void setGrid(final DynamicGrid cooked) {
    this.cooked = cooked;
  }

  public void setBinsCount(final int binsCount) {
    this.binsCount = binsCount;
  }

  public void setDataSet(final VecDataSet dataSet) {
    this.dataSet = dataSet;
  }

  @Override
  public DynamicGrid create() {
    if (cooked == null) {
      cooked = new BFDynamicGrid(dataSet, binsCount);
    }
    return cooked;
  }
}
