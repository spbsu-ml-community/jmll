package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.DataSetImpl;

import java.util.HashMap;
import java.util.Map;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 18:43
 */
public class Binarize implements Computable<DataSet, Binarize> {
  Map<BFGrid, BinarizedDataSet> grids = new HashMap<BFGrid, BinarizedDataSet>(1);
  DataSetImpl set;
  public synchronized BinarizedDataSet binarize(BFGrid grid) {
    BinarizedDataSet result = grids.get(grid);
    if (result == null)
      grids.put(grid, result = new BinarizedDataSet(set, grid));
    return result;
  }

  @Override
  public Binarize compute(DataSet argument) {
    set = (DataSetImpl)argument;
    return this;
  }
}
