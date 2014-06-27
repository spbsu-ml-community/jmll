package com.spbsu.ml.data.stats;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.DataSetImpl;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * Created with IntelliJ IDEA.
 * User: solar
 * Date: 22.03.13
 * Time: 20:32
 * To change this template use File | Settings | File Templates.
 */
public class OrderByFeature implements Computable<DataSet, OrderByFeature> {
  final TIntObjectHashMap<ArrayPermutation> orders = new TIntObjectHashMap<ArrayPermutation>();
  DataSetImpl set;

  @Override
  public OrderByFeature compute(DataSet argument) {
    set = (DataSetImpl)argument;
    return this;
  }

  public synchronized ArrayPermutation orderBy(int featureNo) {
    ArrayPermutation result = orders.get(featureNo);
    if (result == null)
      orders.put(featureNo, result = new ArrayPermutation(set.order(featureNo)));
    return result;
  }
}
