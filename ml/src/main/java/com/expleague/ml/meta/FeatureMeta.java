package com.expleague.ml.meta;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVecBuilder;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.seq.*;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.impl.FeatureMetaImpl;

/**
 * User: solar
 * Date: 16.07.14
 * Time: 18:01
 */
public interface FeatureMeta {
  String id();
  String description();
  ValueType type();

  static FeatureMeta create(String id, String description, ValueType vec) {
    return new FeatureMetaImpl(id, description, vec);
  }


  enum ValueType {
    VEC(Vec.class),
    SPARSE_VEC(SparseVec.class),
    INTS(IntSeq.class),
    VEC_SEQ(VecSeq.class),
    CHAR_SEQ(CharSeq.class);

    private final Class<? extends Seq<?>> type;

    ValueType(final Class<? extends Seq<?>> type) {
      this.type = type;
    }

    public static ValueType fit(Seq<?> target) {
      if (target instanceof SparseVec)
        return SPARSE_VEC;
      else if (target instanceof Vec)
        return VEC;
      else if (target instanceof IntSeq)
        return INTS;
      else if (Vec.class.isAssignableFrom(target.elementType()))
        return VEC_SEQ;
      else if (CharSeq.class.isAssignableFrom(target.elementType()))
        return CHAR_SEQ;
      throw new IllegalArgumentException();
    }

    public Class<? extends Seq<?>> clazz() {
      return type;
    }

    public SeqBuilder<?> builder() {
      if (SparseVec.class.isAssignableFrom(type))
        return new SparseVecBuilder();
      else if (Vec.class.isAssignableFrom(type))
        return new VecBuilder();
      else if (IntSeq.class.isAssignableFrom(type))
        return new IntSeqBuilder();
      else if (CharSeq.class.isAssignableFrom(type))
        return new CharSeqBuilder();
      else if (VecSeq.class.isAssignableFrom(type))
        return new ArraySeqBuilder<>(Vec.class);
      else
        return new ArraySeqBuilder<>(clazz());
    }
  }

  abstract class Stub implements FeatureMeta {
    @Override
    public final boolean equals(final Object o) {
      if (this == o)
        return true;
      if (!(o instanceof FeatureMeta))
        return false;

      final FeatureMeta other = (FeatureMeta) o;
      return id().equals(other.id());
    }

    @Override
    public final int hashCode() {
      return id().hashCode();
    }

    @Override
    public String toString() {
      return id() + ": " + description();
    }
  }
}
