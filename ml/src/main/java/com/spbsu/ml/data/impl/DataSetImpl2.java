package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DSIterator;

/**
 * Created by vkokarev on 08.04.14.
 * this dataSet is the same as DataSetImpl, but has another iterator implementation
 * helps to prevent overflow in some cases
 */
public class DataSetImpl2 extends DataSetImpl {
  public DataSetImpl2(final double[] data, final double[] target) {
    super(data, target);
  }

  public DataSetImpl2(final Mx data, final Vec target) {
    super(data, target);
  }

  @Override
  public DSIterator iterator() {
    return new DSIterator() {
      private final Mx data = DataSetImpl2.this.data();
      private final Vec target = DataSetImpl2.this.target();
      private final int maxRow = data.rows();
      private int cRow = -1;

      @Override
      public boolean advance() {
        return ++cRow < maxRow;
      }

      @Override
      public double y() {
        return target.get(cRow);
      }

      @Override
      public double x(final int i) {
        return data.get(cRow, i);
      }

      @Override
      public Vec x() {
        return data.row(cRow);
      }

      @Override
      public int index() {
        return cRow;
      }
    };
  }
}
