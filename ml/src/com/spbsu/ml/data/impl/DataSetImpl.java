package com.spbsu.ml.data.impl;

import com.spbsu.commons.func.CacheHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:36
 */
public class DataSetImpl extends CacheHolderImpl implements DataSet {
  protected final Mx data;
  protected final Vec target;

  public DataSetImpl(double[] data, double[] target) {
    this.data = new VecBasedMx(data.length / target.length, new ArrayVec(data));
    this.target = new ArrayVec(target);
  }

  public DataSetImpl(Mx data, Vec target) {
    this.data = data;
    this.target = target;
  }

  public int power() {
    return target.dim();
  }

  public int xdim() {
    return data.columns();
  }

  public DSIterator iterator() {
    return new DSIterator() {
      Mx data = DataSetImpl.this.data();
      Vec target = DataSetImpl.this.target();
      final int step = xdim();
      int index = -step;

      public boolean advance() {
        return (index+=step) < data.dim();
      }

      public double y() {
        return target.get(index/step);
      }

      public double x(int i) {
        return data.get(index + i);
      }

      public Vec x() {
        return data.row(index/step);
      }

      @Override
      public int index() {
        return index/step;
      }
    };
  }

  public int[] order(final int featureIndex) {
    final int[] result = ArrayTools.sequence(0, power());
    ArrayTools.parallelSort(data.col(featureIndex).toArray(), result);
    return result;
  }

  public Vec target() {
    return target;
  }

  public Mx data() {
    return data;
  }

}
