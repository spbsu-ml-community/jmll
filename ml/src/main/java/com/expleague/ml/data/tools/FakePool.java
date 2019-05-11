package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.Vectorization;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.meta.*;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;
import com.expleague.ml.meta.impl.fake.FakeTargetMeta;
import com.expleague.ml.meta.items.FakeItem;
import com.fasterxml.jackson.annotation.JsonIgnore;

import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Objects;
import java.util.function.IntFunction;
import java.util.stream.Stream;

/**
 * User: solar
 * Date: 18.07.14
 * Time: 18:40
 */
public class FakePool<T extends FakeItem> extends Pool<T> {
  private final Mx data;

  public static FakePool<FakeItem> create(final Mx data, final Seq<?> target) {
    return new FakePool<>(data, target, FakePool::genItems);
  }

  protected FakePool(final Mx data, final Seq<?> target, IntFunction<Seq<T>> supplier) {
    this(
        new JsonDataSetMeta("features.txt", "/dev/random", new Date(), FakeItem.class, "dsitems"),
        supplier.apply(target.length()),
        genFakeFeatures(data, target), data
    );
    Stream.of(features()).forEach(f -> f.setOwner(this));
  }

  protected FakePool(DataSetMeta meta, Seq<T> items, LinkedHashMap<PoolFeatureMeta, Seq<?>> features, Mx data) {
    super(meta, items, features);
    this.data = data;
  }

  private static LinkedHashMap<PoolFeatureMeta, Seq<?>> genFakeFeatures(final Mx data, Seq<?> target) {
    final LinkedHashMap<PoolFeatureMeta, Seq<?>> features = new LinkedHashMap<>();
    for (int i = 0; i < data.columns(); i++) {
      final PoolFeatureMeta.ValueType type = VecTools.isSparse(data.col(i), 0.1) ? PoolFeatureMeta.ValueType.SPARSE_VEC : PoolFeatureMeta.ValueType.VEC;
      final JsonFeatureMeta meta = new JsonFeatureMeta("Fake-" + i, "Fake feature from features.txt format #" + i, type);
      features.put(meta, type == PoolFeatureMeta.ValueType.VEC ? data.col(i) : VecTools.copySparse(data.col(i)));
    }
    features.put(new JsonTargetMeta("fake-target", "Target from features.txt format", FeatureMeta.ValueType.fit(target)), target);
    return features;
  }

  @Override
  public VecDataSet vecData() {
    final DataSet<T> ds = data();
    return new VecDataSetImpl(ds, data, new Vectorization<T>() {
      @Override
      public Vec value(final T subject) {
        return data.row(ds.index(subject));
      }

      @Override
      public FeatureMeta meta(final int findex) {
        return fmeta(findex);
      }

      @Override
      public int dim() {
        return data.columns();
      }
    });
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (!(o instanceof FakePool))
      return false;
    if (!super.equals(o))
      return false;
    FakePool<?> fakePool = (FakePool<?>) o;
    return data.equals(fakePool.data);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), data);
  }

  public static Seq<FakeItem> genItems(final int count) {
    final FakeItem[] result = new FakeItem[count];
    for (int i = 0; i < result.length; i++)
      result[i] = new FakeItem(i);
    return new ArraySeq<>(result);
  }
}
