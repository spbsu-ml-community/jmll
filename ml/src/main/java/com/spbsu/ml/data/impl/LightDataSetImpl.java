package com.spbsu.ml.data.impl;


import com.spbsu.commons.func.CacheHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.VectorizedRealTargetDataSet;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolMeta;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:36
 */
public class LightDataSetImpl extends CacheHolderImpl implements VectorizedRealTargetDataSet<Integer> {
  protected final Mx data;
  protected final Vec target;

  public LightDataSetImpl(double[] data, double[] target) {
    this.data = new VecBasedMx(data.length / target.length, new ArrayVec(data));
    this.target = new ArrayVec(target);
  }

  public LightDataSetImpl(Mx data, Vec target) {
    this.data = data;
    this.target = target;
  }

  @Override
  public Vec at(final int i) {
    return data().row(i);
  }

  @Override
  public LightDataSetImpl sub(final int start, final int end) {
    return new LightDataSetImpl(data.sub(start, 0, end - start, data.columns()), target.sub(start, end));
  }

  public int length() {
    return target.dim();
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public DataSet<Integer> original() {
    return new Stub<Integer>() {
      @Override
      public Integer at(final int i) {
        return i;
      }

      @Override
      public int length() {
        return target.dim();
      }
    };
  }

  @Override
  public Vectorization<Integer> vectorization() {
    return new Vectorization<Integer>() {
      @Override
      public Vec value(final Integer subject) {
        return data().row(subject);
      }

      @Override
      public FeatureMeta meta(final int findex) {
        return new FeatureMeta() {
          @Override
          public String id() {
            return "fake-" + findex;
          }

          @Override
          public String description() {
            return "This is FAKE feature";
          }
        };
      }

      @Override
      public int dim() {
        return xdim();
      }
    };
  }

  @Override
  public double target(final Integer x) {
    return target().get(x);
  }

  public int xdim() {
    return data.columns();
  }

  public int[] order(final int featureIndex) {
    final int[] result = ArrayTools.sequence(0, length());
    ArrayTools.parallelSort(data.col(featureIndex).toArray(), result);
    return result;
  }

  public Vec target() {
    return target;
  }

  public Mx data() {
    return data;
  }

  @Override
  public PoolMeta meta() {
    return null;
  }
}
