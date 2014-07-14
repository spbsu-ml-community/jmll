package com.spbsu.ml.data;

import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:34
 */
public interface StatisticCalculator {
    public double value(DataSet set);
}
