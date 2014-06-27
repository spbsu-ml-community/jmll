package com.spbsu.ml.data.stats;

import com.spbsu.commons.func.Computable;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:13
 */
public class TotalStat implements Computable<DataSet, Double> {
  @Override
  public Double compute(DataSet set) {
    final DSIterator iter = set.iterator();
    double sum = 0;
    while(iter.advance()) {
        sum += iter.y();
    }
    return sum;
  }
}
