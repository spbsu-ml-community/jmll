package com.expleague.ml;

import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 18:43
 */
public class Binarize implements Function<VecDataSet, Binarize> {
  private final Map<BFGrid, BinarizedDataSet> grids = new HashMap<>(1);
  private final Map<DynamicGrid, BinarizedDynamicDataSet> dynamicGrids = new HashMap<>(1);
  private VecDataSet set;

  public synchronized BinarizedDataSet binarize(final BFGrid grid) {
    return grids.compute(grid, (key, value) -> value != null ? value : new BinarizedDataSet(set, grid));
  }

  public synchronized BinarizedDynamicDataSet binarize(final DynamicGrid grid) {
    return dynamicGrids.compute(grid, (key, value) -> value != null ? value : new BinarizedDynamicDataSet(set, grid));
  }

  @Override
  public Binarize apply(final VecDataSet argument) {
    set = argument;
    return this;
  }
}
