package ml.data.impl;

import ml.data.DataSet;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:55
 */
public interface DataSetInner extends DataSet {
    double[] data();
    double[] target();
}
