package com.expleague.ml.dynamicGrid.models;

import com.expleague.commons.math.Func;
import com.expleague.ml.dynamicGrid.impl.BinarizedDynamicDataSet;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface BinDynamicOptimizedModel extends Func {
  double value(BinarizedDynamicDataSet bds, int index);
}
