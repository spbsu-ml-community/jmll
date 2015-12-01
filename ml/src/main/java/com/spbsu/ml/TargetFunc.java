package com.spbsu.ml;

import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 15.07.14
 * Time: 10:37
 */
public interface TargetFunc extends Func {
  DataSet<?> owner();
}
