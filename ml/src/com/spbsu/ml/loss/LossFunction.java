package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Model;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:25:39
 */
public interface LossFunction {
    Vec gradient(Vec point, DataSet learn);
    double value(Model model, DataSet set);
}
