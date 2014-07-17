package com.spbsu.ml.data.tools;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
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
import com.spbsu.ml.meta.items.QURLItem;

/**
* User: solar
* Date: 07.07.14
* Time: 20:55
*/
public class FeaturesTxtPool extends Pool<QURLItem> {
  private final Mx data;

  public FeaturesTxtPool(final String file, Seq<QURLItem> items, final Mx data, Vec target) {
    super(new FakeDataSetMeta(file), items, genFakeFeatures(data), Pair.create(new FakeTargetMeta(), target));
    this.data = data;
    for (int i = 0; i < features.length; i++) {
      Pair<? extends FeatureMeta, ? extends Seq<?>> feature = features[i];
      ((FakeFeatureMeta)feature.first).owner = this;
    }
    ((FakeTargetMeta)this.target.first).owner = this;
    ((FakeDataSetMeta)meta).owner = this;
  }

  private static Pair<PoolFeatureMeta, Vec>[] genFakeFeatures(final Mx data) {
    List<Pair<PoolFeatureMeta, Vec>> features = new ArrayList<>();
    for (int i = 0; i < data.columns(); i++) {
      final int finalI = i;
      final PoolFeatureMeta.ValueType type = VecTools.isSparse(data.col(i), 0.1) ? PoolFeatureMeta.ValueType.SPARSE_VEC : PoolFeatureMeta.ValueType.VEC;
      features.add(Pair.<PoolFeatureMeta, Vec>create(new FakeFeatureMeta(finalI, type), type == PoolFeatureMeta.ValueType.VEC ? data.col(i) : VecTools.copySparse(data.col(i))));
    }
    //noinspection unchecked
    return features.toArray(new Pair[features.size()]);
  }


  public VecDataSet vecData() {
    final DataSet<QURLItem> ds = data();
    return new VecDataSetImpl(ds, data, new Vectorization<QURLItem>() {
      @Override
      public Vec value(final QURLItem subject) {
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

  private static class FakeFeatureMeta implements PoolFeatureMeta {
    private final int finalI;
    private final ValueType type;
    private Pool<QURLItem> owner;

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
    public DataSet<QURLItem> associated() {
      return owner.data();
    }
  }

  private static class FakeTargetMeta implements TargetMeta {
    private Pool<QURLItem> owner;
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
    public DataSet<QURLItem> associated() {
      return owner.data();
    }
  }

  private static class FakeDataSetMeta implements DataSetMeta {
    private final String file;
    protected Date creationDate;
    private Pool owner;

    public FakeDataSetMeta(final String file) {
      this.file = file;
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
    public ItemType type() { return ItemType.QURL; }

    @Override
    public Date created() {
      return creationDate;
    }
  }
}
