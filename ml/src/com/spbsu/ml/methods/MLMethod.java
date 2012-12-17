package com.spbsu.ml.methods;

import com.spbsu.ml.Model;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.LossFunction;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface MLMethod {
    Model fit(DataSet learn, LossFunction loss);
}
