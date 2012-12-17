package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DataSet;

import java.util.Random;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 18:05
 */
public class Bootstrap extends DataSetImpl {
  private final DataSet original;
  private final int[] order;

  public Bootstrap(DataSet parent, int[] order) {
    super(new VecBasedMx(parent.xdim(), new IndexTransVec(parent.data(), new RowsPermutation(order, parent.xdim()), parent.data().basis())),
          new IndexTransVec(parent.target(),new ArrayPermutation(order), parent.target().basis()));
    original = parent;
    this.order = order;
  }

  public Bootstrap(DataSet parent, Random rnd) {
    this(parent, new int[parent.data().rows()]);
    for (int i = 0; i < order.length; i++) {
      order[i] = rnd.nextInt(order.length);
    }
  }

  public Bootstrap(DataSet parent) {
    this(parent, new FastRandom());
  }

  public int[] order() {
    return order;
  }

  public DataSet original() {
    return original;
  }
}
