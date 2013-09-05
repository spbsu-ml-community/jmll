package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface MLMethodOrder1 {
    Model fit(DataSet learn, Oracle1 loss);
    Model fit(DataSet learn, Oracle1 loss, Vec start);
}
