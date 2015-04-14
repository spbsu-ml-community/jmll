package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.ml.data.impl.RankedDataSet;
import com.spbsu.ml.data.set.VecDataSet;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 18:43
 */
public class RankIt implements Computable<VecDataSet, RankIt> {
  VecDataSet set;
  RankedDataSet result = null;

  public synchronized RankedDataSet value() {
    if (result == null)
      result = new RankedDataSet(set);
    return result;
  }

  @Override
  public RankIt compute(final VecDataSet argument) {
    set = argument;
    return this;
  }
}
