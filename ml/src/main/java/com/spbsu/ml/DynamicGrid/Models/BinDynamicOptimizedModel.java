package com.spbsu.ml.DynamicGrid.Models;

import com.spbsu.ml.DynamicGrid.Impl.BinarizedDynamicDataSet;
import com.spbsu.ml.Func;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface BinDynamicOptimizedModel extends Func {
  double value(BinarizedDynamicDataSet bds, int index);
}
