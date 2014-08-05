package com.spbsu.ml.dynamicGridFix.modelsFix;

import com.spbsu.ml.Func;
import com.spbsu.ml.dynamicGridFix.implFix.BinarizedDynamicDataSet;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface BinDynamicOptimizedModel extends Func {
  double value(BinarizedDynamicDataSet bds, int index);
}
