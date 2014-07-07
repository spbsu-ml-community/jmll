package com.spbsu.ml.data.stats;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.ml.data.VectorizedRealTargetDataSet;
import com.spbsu.ml.data.impl.LightDataSetImpl;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * Created with IntelliJ IDEA.
 * User: solar
 * Date: 22.03.13
 * Time: 20:32
 * To change this template use File | Settings | File Templates.
 */
public class OrderByFeature implements Computable<VectorizedRealTargetDataSet, OrderByFeature> {
  final TIntObjectHashMap<ArrayPermutation> orders = new TIntObjectHashMap<ArrayPermutation>();
  LightDataSetImpl set;

  @Override
  public OrderByFeature compute(VectorizedRealTargetDataSet argument) {
    set = (LightDataSetImpl)argument;
    return this;
  }

  public synchronized ArrayPermutation orderBy(int featureNo) {
    ArrayPermutation result = orders.get(featureNo);
    if (result == null)
      orders.put(featureNo, result = new ArrayPermutation(set.order(featureNo)));
    return result;
  }
}
