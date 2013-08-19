package com.spbsu.ml.data.impl;

import com.spbsu.commons.func.CacheHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import gnu.trove.TIntObjectHashMap;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:36
 */
public class DataSetImpl extends CacheHolderImpl implements DataSet {
  private final Mx data;
  private final Vec target;

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
    };
  }

  final TIntObjectHashMap<int[]> orders = new TIntObjectHashMap<int[]>();
  public synchronized int[] order(final int featureIndex) {
    int[] result = orders.get(featureIndex);
    if (result == null) {
      result = ArrayTools.sequence(0, power());
      ArrayTools.parallelSort(data.col(featureIndex).toArray(), result);
      orders.put(featureIndex, result);
    }
    return result;
  }

  public Vec target() {
    return target;
  }

  public Mx data() {
    return data;
  }

}
