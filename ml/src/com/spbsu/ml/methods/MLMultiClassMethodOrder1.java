package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.MultiClassModel;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface MLMultiClassMethodOrder1 extends MLMethodOrder1 {
  MultiClassModel fit(DataSet learn, Oracle1 loss);
  MultiClassModel fit(DataSet learn, Oracle1 loss, Vec[] start);
}
