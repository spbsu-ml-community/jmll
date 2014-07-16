package com.spbsu.ml.data.tools;

import java.util.ArrayList;
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
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolMeta;
import com.spbsu.ml.meta.TargetMeta;
import com.spbsu.ml.meta.items.QURLItem;

/**
* User: solar
* Date: 07.07.14
* Time: 20:55
*/
public class FeaturesTxtPool extends Pool<QURLItem> {
  private final Mx data;

  public FeaturesTxtPool(final String file, Seq<QURLItem> items, Mx data, Vec target) {
    super(new PoolMeta.FakePoolMeta() {
      @Override
      public String file() {
        return file;
      }
    }, items, genFakeFeatures(data), Pair.create(new TargetMeta() {
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
    }, target));
    this.data = data;
  }

  private static Pair<FeatureMeta, Vec>[] genFakeFeatures(final Mx data) {
    List<Pair<FeatureMeta, Vec>> features = new ArrayList<>();
    for (int i = 0; i < data.columns(); i++) {
      final int finalI = i;
      final FeatureMeta.ValueType type = VecTools.isSparse(data.col(i), 0.1) ? FeatureMeta.ValueType.SPARSE_VEC : FeatureMeta.ValueType.VEC;
      features.add(Pair.<FeatureMeta, Vec>create(new FeatureMeta() {
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
      }, type == FeatureMeta.ValueType.VEC ? data.col(i) : VecTools.copySparse(data.col(i))));
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
}
