package com.spbsu.ml.data.stats;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.DataSetImpl;
import gnu.trove.TIntObjectHashMap;

/**
 * Created with IntelliJ IDEA.
 * User: solar
 * Date: 22.03.13
 * Time: 20:32
 * To change this template use File | Settings | File Templates.
 */
public class OrderByFeature  implements Computable<DataSet, OrderByFeature> {
  final TIntObjectHashMap<int[]> orders = new TIntObjectHashMap<int[]>();
  DataSetImpl set;

  @Override
  public OrderByFeature compute(DataSet argument) {
    set = (DataSetImpl)argument;
    return this;
  }

  public ArrayPermutation orderBy(int features) {
    return new ArrayPermutation(set.order(features));
  }
}
