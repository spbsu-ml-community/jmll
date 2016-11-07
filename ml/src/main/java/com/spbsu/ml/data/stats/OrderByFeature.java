package com.spbsu.ml.data.stats;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
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
  VecDataSet set;

  @Override
  public OrderByFeature compute(final DataSet argument) {
    set = (VecDataSet)argument;
    return this;
  }

  private int[] order(int featureIndex) {
    final int[] result = ArrayTools.sequence(0, set.length());
    ArrayTools.parallelSort(set.data().col(featureIndex).toArray(), result);
    return result;
  }
  public synchronized ArrayPermutation orderBy(final int featureNo) {
    ArrayPermutation result = orders.get(featureNo);
    if (result == null)
      orders.put(featureNo, result = new ArrayPermutation(order(featureNo)));
    return result;
  }
}
