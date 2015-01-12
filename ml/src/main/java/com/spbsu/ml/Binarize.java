package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;

import java.util.HashMap;
import java.util.Map;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 18:43
 */
public class Binarize implements Computable<VecDataSet, Binarize> {
  Map<BFGrid, BinarizedDataSet> grids = new HashMap<>(1);
  Map<DynamicGrid, BinarizedDynamicDataSet> dynamicGrids = new HashMap<>(1);
  VecDataSet set;

  public synchronized BinarizedDataSet binarize(final BFGrid grid) {
    BinarizedDataSet result = grids.get(grid);
    if (result == null)
      grids.put(grid, result = new BinarizedDataSet(set, grid));
    return result;
  }

  public synchronized BinarizedDynamicDataSet binarize(final DynamicGrid grid) {
    BinarizedDynamicDataSet result = dynamicGrids.get(grid);
    if (result == null)
      dynamicGrids.put(grid, result = new BinarizedDynamicDataSet(set, grid));
    return result;
  }

  @Override
  public Binarize compute(final VecDataSet argument) {
    set = argument;
    return this;
  }
}
