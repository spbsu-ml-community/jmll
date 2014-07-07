package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.ml.data.VectorizedRealTargetDataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.LightDataSetImpl;


import java.util.HashMap;
import java.util.Map;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 18:43
 */
public class Binarize implements Computable<VectorizedRealTargetDataSet, Binarize> {
  Map<BFGrid, BinarizedDataSet> grids = new HashMap<BFGrid, BinarizedDataSet>(1);
  LightDataSetImpl set;
  public synchronized BinarizedDataSet binarize(BFGrid grid) {
    BinarizedDataSet result = grids.get(grid);
    if (result == null)
      grids.put(grid, result = new BinarizedDataSet(set, grid));
    return result;
  }

  @Override
  public Binarize compute(VectorizedRealTargetDataSet argument) {
    set = (LightDataSetImpl)argument;
    return this;
  }
}
