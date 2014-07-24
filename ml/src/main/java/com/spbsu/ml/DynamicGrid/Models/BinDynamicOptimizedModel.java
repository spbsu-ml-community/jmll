package com.spbsu.ml.DynamicGrid.Models;

import com.spbsu.ml.Func;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.BinarizedDynamicDataSet;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface BinDynamicOptimizedModel extends Func {
    double value(BinarizedDynamicDataSet bds, int index);
}
