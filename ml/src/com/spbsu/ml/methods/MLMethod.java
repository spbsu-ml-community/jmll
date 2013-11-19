package com.spbsu.ml.methods;

import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle0;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface MLMethod<Loss extends Oracle0> {
  Model fit(DataSet learn, Loss loss);
}
