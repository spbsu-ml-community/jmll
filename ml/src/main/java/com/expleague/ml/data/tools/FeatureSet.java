package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.ml.data.tools.impl.JoinedFeatureSet;
import com.expleague.ml.meta.FeatureMeta;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public interface FeatureSet {
  Vec advance();
  Vec advanceTo(Vec v);
  Vec advanceTo(Vec v, FeatureMeta... features);
  int dim();
  FeatureMeta meta(int index);

  static FeatureSet join(FeatureSet... fs) {
    return new JoinedFeatureSet(fs);
  }

  abstract class Stub implements FeatureSet {
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
      VecTools.assign(current, to);
      return to;
    }

    @Override
    public Vec advanceTo(Vec to, FeatureMeta... features) {
      advanceTo(current);
      for (int i = 0; i < features.length; i++) {
        to.set(i, current.get(metaIndex.get(features[i])));
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
  }

  class Accumulator {
    private final FeatureSet fs;
    private final List<VecBuilder> builders;

    public Accumulator(FeatureSet fs) {
      this.fs = fs;
      this.builders = new ArrayList<>(fs.dim());
      for (int i = 0; i < fs.dim(); i++) {
        this.builders.add(new VecBuilder());
      }
    }

    public void advance() {
      final Vec vec = fs.advance();
      for (int i = 0; i < fs.dim(); i++) {
        builders.get(i).append(vec.get(i));
      }
    }

    public void splashOut(PoolBuilder builder) {
      for (int i = 0; i < fs.dim(); i++) {
        builder.newFeature(fs.meta(i), builders.get(i).build());
      }
    }
  }
}
