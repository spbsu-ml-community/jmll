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
import com.expleague.ml.meta.impl.fake.FakeTargetMeta;
import com.expleague.ml.meta.items.FakeItem;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.Date;
import java.util.LinkedHashMap;
import java.util.function.IntFunction;

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
    this(new FakeDataSetMeta(), supplier.apply(target.length()), genFakeFeatures(data, target), data);
  }

  protected FakePool(DataSetMeta meta, Seq<T> items, LinkedHashMap<PoolFeatureMeta, Seq<?>> features, Mx data) {
    super(meta, items, features);
    this.data = data;
  }

  private static LinkedHashMap<PoolFeatureMeta, Seq<?>> genFakeFeatures(final Mx data, Seq<?> target) {
    final LinkedHashMap<PoolFeatureMeta, Seq<?>> features = new LinkedHashMap<>();
    for (int i = 0; i < data.columns(); i++) {
      final PoolFeatureMeta.ValueType type = VecTools.isSparse(data.col(i), 0.1) ? PoolFeatureMeta.ValueType.SPARSE_VEC : PoolFeatureMeta.ValueType.VEC;
      features.put(new FakeFeatureMeta(i, type), type == PoolFeatureMeta.ValueType.VEC ? data.col(i) : VecTools.copySparse(data.col(i)));
    }
    features.put(new FakeTargetMeta(FeatureMeta.ValueType.fit(target), 0), target);
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
  public boolean equals(final Object obj) {
    if (obj == this)
      return true;
    if (obj == null || obj.getClass() != getClass())
      return false;

    final FakePool other = (FakePool) obj;
    return new EqualsBuilder()
        .appendSuper(super.equals(obj))
        .append(data, other.data)
        .isEquals();
  }

  @Override
  public int hashCode() {
    return new HashCodeBuilder().appendSuper(super.hashCode()).append(data).toHashCode();
  }

  private static class FakeFeatureMeta implements PoolFeatureMeta {
    private final int finalI;
    private final ValueType type;
    private Pool owner;

    public FakeFeatureMeta(final int finalI, final ValueType type) {
      this.finalI = finalI;
      this.type = type;
    }

    @Override
    public String id() {
      return "Fake-" + finalI;
    }

    @Override
    public String description() {
      return "Fake feature from features.txt format #" + finalI;
    }

    @Override
    public ValueType type() {
      return type;
    }

    @Override
    public <T extends DSItem> Pool<T> owner() {
      //noinspection unchecked
      return (Pool<T>) owner;
    }

    @Override
    public void setOwner(Pool<? extends DSItem> owner) {
      this.owner = owner;
    }

    @Override
    public DataSet<FakeItem> associated() {
      return owner.data();
    }
  }


  private static class FakeDataSetMeta implements DataSetMeta {
    protected Date creationDate;
    private Pool owner;

    public FakeDataSetMeta() {
      creationDate = new Date(0);
    }

    @Override
    public String id() {
      return "dsitems";
    }

    @Override
    public String source() {
      return "/dev/random";
    }

    @Override
    public String author() {
      return "/dev/null";
    }

    @Override
    public Pool owner() {
      return owner;
    }

    @Override
    public void setOwner(Pool pool) {
      this.owner = pool;
    }

    @Override
    public Class<?> type() { return FakeItem.class; }

    @Override
    public Date created() {
      return creationDate;
    }
  }

  public static Seq<FakeItem> genItems(final int count) {
    final FakeItem[] result = new FakeItem[count];
    for (int i = 0; i < result.length; i++)
      result[i] = new FakeItem(i);
    return new ArraySeq<>(result);
  }
}
