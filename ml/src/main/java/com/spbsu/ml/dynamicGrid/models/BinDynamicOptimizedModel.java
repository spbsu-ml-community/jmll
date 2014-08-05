package com.spbsu.ml.dynamicGrid.models;

import com.spbsu.ml.Func;
import com.spbsu.ml.dynamicGrid.impl.BinarizedDynamicDataSet;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface BinDynamicOptimizedModel extends Func {
  double value(BinarizedDynamicDataSet bds, int index);
}
