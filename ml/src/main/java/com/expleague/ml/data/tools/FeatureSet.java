package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.ml.data.tools.impl.JoinedFeatureSet;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public interface FeatureSet<T extends DSItem> extends Consumer<T> {
  void accept(T item);
  Vec advance();
  Vec advanceTo(Vec v);
  Vec advanceTo(Vec v, FeatureMeta... features);
  int dim();
  FeatureMeta meta(int index);

  @SafeVarargs
  static <T extends DSItem> FeatureSet<T> join(FeatureSet<? super T>... fs) {
    return new JoinedFeatureSet<>(fs);
  }

  int index(FeatureMeta meta);

  Stream<FeatureSet<? super T>> components();

  abstract class Stub<T extends DSItem> implements FeatureSet<T> {
    private TObjectIntMap<FeatureMeta> metaIndex;
    private FeatureMeta[] metas;
    private BitSet assigned;

    private Vec current;

    protected Stub() {
      init(Stream.of(getClass().getFields())
          .filter(fld -> (fld.getModifiers() & Modifier.STATIC) != 0)
          .map(fld -> fld.getType().isAssignableFrom(FeatureMeta.class) ? fld : null)
          .filter(Objects::nonNull)
          .toArray(FeatureMeta[]::new));
    }

    protected Stub(FeatureMeta... metas) {
      init(metas);
    }

    @Override
    public void accept(T item) {
    }

    private void init(FeatureMeta[] metas) {
      this.metas = metas;
      metaIndex = new TObjectIntHashMap<>(metas.length * 2, 0.7f, -1);
      IntStream.range(0, metas.length).forEach(idx -> metaIndex.put(metas[idx], idx));
      current = new ArrayVec(metas.length);
      assigned = new BitSet(metas.length);
    }

    protected void set(FeatureMeta meta, double value) {
      final int idx = metaIndex.get(meta);
      assigned.set(idx);
      current.set(idx, value);
    }

    @Override
    public Vec advanceTo(Vec to) {
      if (assigned.cardinality() < metas.length)
        throw new IllegalStateException("Not all features are set " + assigned);
      assigned.clear();
      VecTools.assign(to, current);
      return to;
    }

    @Override
    public Vec advanceTo(Vec to, FeatureMeta... features) {
      advanceTo(current);
      for (int i = 0; i < features.length; i++) {
        to.set(i, current.get(index(features[i])));
      }
      return to;
    }

    @Override
    public Vec advance() {
      return advanceTo(new ArrayVec(dim()));
    }

    @Override
    public int dim() {
      return metas.length;
    }

    @Override
    public FeatureMeta meta(int index) {
      return metas[index];
    }

    @Override
    public int index(FeatureMeta meta) {
      return this.metaIndex.get(meta);
    }

    @Override
    public Stream<FeatureSet<? super T>> components() {
      return Stream.of(this);
    }
  }
}
