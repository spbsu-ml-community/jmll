package com.spbsu.ml.meta;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;

/**
 * User: solar
 * Date: 16.07.14
 * Time: 18:01
 */
public interface FeatureMeta {
  String id();

  String description();

  ValueType type();

  enum ValueType {
    VEC(Vec.class),
    SPARSE_VEC(SparseVec.class),
    INTS(IntSeq.class);

    private final Class<? extends Seq<?>> type;

    private ValueType(Class<? extends Seq<?>> type) {
      this.type = type;
    }

    public Class<? extends Seq<?>> clazz() {
      return type;
    }
  }
}
