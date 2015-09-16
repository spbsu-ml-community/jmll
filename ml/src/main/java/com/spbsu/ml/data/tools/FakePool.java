package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.meta.DataSetMeta;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolFeatureMeta;
import com.spbsu.ml.meta.TargetMeta;
import com.spbsu.ml.meta.items.FakeItem;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * User: solar
 * Date: 18.07.14
 * Time: 18:40
 */
public class FakePool extends Pool<FakeItem> {
  private final Mx data;

  public FakePool(final Mx data, final Seq<?> target) {
    super(new FakeDataSetMeta(), genItems(target.length()), genFakeFeatures(data), new Pair[]{Pair.create(new FakeTargetMeta(), target)});
    this.data = data;
    for (int i = 0; i < features.length; i++) {
      final Pair<? extends FeatureMeta, ? extends Seq<?>> feature = features[i];
      ((FakeFeatureMeta)feature.first).owner = this;
    }
    ((FakeTargetMeta)this.targets.get(0).first).owner = this;
    ((FakeDataSetMeta)meta).owner = this;
  }

  private static Pair<PoolFeatureMeta, Vec>[] genFakeFeatures(final Mx data) {
    final List<Pair<PoolFeatureMeta, Vec>> features = new ArrayList<>();
    for (int i = 0; i < data.columns(); i++) {
      final int finalI = i;
      final PoolFeatureMeta.ValueType type = VecTools.isSparse(data.col(i), 0.1) ? PoolFeatureMeta.ValueType.SPARSE_VEC : PoolFeatureMeta.ValueType.VEC;
      features.add(Pair.<PoolFeatureMeta, Vec>create(new FakeFeatureMeta(finalI, type), type == PoolFeatureMeta.ValueType.VEC ? data.col(i) : VecTools.copySparse(data.col(i))));
    }
    //noinspection unchecked
    return features.toArray(new Pair[features.size()]);
  }

  @Override
  public VecDataSet vecData() {
    final DataSet<FakeItem> ds = data();
    return new VecDataSetImpl(ds, data, new Vectorization<FakeItem>() {
      @Override
      public Vec value(final FakeItem subject) {
        return data.row(ds.index(subject));
      }

      @Override
      public FeatureMeta meta(final int findex) {
        return features[findex].first;
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
    private Pool<FakeItem> owner;

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
    public DataSet<FakeItem> associated() {
      return owner.data();
    }
  }

  private static class FakeTargetMeta implements TargetMeta {
    private Pool<FakeItem> owner;
    @Override
    public String id() {
      return "whoknowsthefakeid";
    }

    @Override
    public String description() {
      return "fake relevance marks";
    }

    @Override
    public ValueType type() {
      return ValueType.VEC;
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
      return "qurls";
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
    public ItemType type() { return ItemType.FAKE; }

    @Override
    public Date created() {
      return creationDate;
    }
  }

  private static Seq<FakeItem> genItems(final int count) {
    final FakeItem[] result = new FakeItem[count];
    for (int i = 0; i < result.length; i++)
      result[i] = new FakeItem(i);
    return new ArraySeq<FakeItem>(result);
  }
}
