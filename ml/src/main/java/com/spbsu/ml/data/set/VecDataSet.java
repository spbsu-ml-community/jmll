package com.spbsu.ml.data.set;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:36
 */
public interface VecDataSet extends DataSet<Vec> {
  Mx data();
  int xdim();
  int[] order(final int featureIndex);
  Vectorization<?> vec();
  FeatureMeta fmeta(int findex);

  abstract class Stub extends DataSet.Stub<Vec> implements VecDataSet {
    private final Vectorization<?> vec;

    public Stub(final VecDataSet parent) {
      super(parent);
      vec = parent != null ? parent.vec() : null;
    }

    public Stub(final DataSet<?> parent, final Vectorization<?> vec) {
      super(parent);
      this.vec = vec;
    }

    @Override
    public final Vec at(final int i) {
      return data().row(i);
    }

    @Override
    public final int length() {
      return data().rows();
    }

    @Override
    public final int xdim() {
      return data().columns();
    }

    @Override
    public int[] order(final int featureIndex) {
      final int[] result = ArrayTools.sequence(0, length());
      ArrayTools.parallelSort(data().col(featureIndex).toArray(), result);
      return result;
    }

    @Override
    public Vectorization<?> vec() {
      return vec;
    }

    @Override
    public final FeatureMeta fmeta(final int findex) {
      return vec().meta(findex);
    }

    @Override
    public Class<Vec> elementType() {
      return Vec.class;
    }
  }
}
