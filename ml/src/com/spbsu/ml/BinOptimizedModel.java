package com.spbsu.ml;

import com.spbsu.ml.data.impl.BinarizedDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface BinOptimizedModel extends Func {
  double value(BinarizedDataSet bds, int index);
}
