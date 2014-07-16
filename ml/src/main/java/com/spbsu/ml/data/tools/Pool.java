package com.spbsu.ml.data.tools;

import java.util.ArrayList;
import java.util.List;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolMeta;
import com.spbsu.ml.meta.TargetMeta;
import gnu.trove.map.hash.TObjectIntHashMap;

/**
* User: solar
* Date: 07.07.14
* Time: 20:55
*
*/
public class Pool<I extends DSItem> {
  protected final PoolMeta meta;
  protected final Pair<? extends TargetMeta, ? extends Seq<?>> target;
  protected final Seq<I> items;
  protected final Pair<? extends FeatureMeta, ? extends Seq<?>>[] features;

  public Pool(final PoolMeta meta, final Seq<I> items, final Pair<? extends FeatureMeta, ? extends Seq<?>>[] features, final Pair<? extends TargetMeta, ? extends Seq<?>> target) {
    this.meta = meta;
    this.target = target;
    this.items = items;
    this.features = features;
  }

  public PoolMeta meta() {
    return meta;
  }

  public DataSet<I> data() {
    final TObjectIntHashMap<I> indices = new TObjectIntHashMap<>();
    for (int i = 0; i < items.length(); i++) {
      indices.put(items.at(i), i);
    }
    return new DataSet.Stub<I>(null) {
      @Override
      public I at(final int i) {
        return items.at(i);
      }

      @Override
      public int length() {
        return items.length();
      }

      @Override
      public PoolMeta meta() {
        return meta;
      }

      @Override
      public int index(final I obj) {
        return indices.get(obj);
      }
    };
  }

  private VecDataSet joinFeatures(final int[] indices) {
    final List<Vec> cols = new ArrayList<>();
    for (int i = 0; i < indices.length; i++) {
      cols.add((Vec)features[indices[i]].second);
    }

    final Mx data = new ColsVecArrayMx(cols.toArray(new Vec[cols.size()]));
    final DataSet<I> ds = data();
    return new VecDataSetImpl(ds, data, new Vectorization<I>() {
      @Override
      public Vec value(final I subject) {
        return data.row(ds.index(subject));
      }

      @Override
      public FeatureMeta meta(final int findex) {
        return features[indices[findex]].first;
      }

      @Override
      public int dim() {
        return indices.length;
      }
    });
  }

  public VecDataSet vecData() {
    return joinFeatures(ArrayTools.sequence(0, features.length));
  }

  public <T extends TargetFunc> T target(Class<T> targetClass) {
    return DataTools.newTarget(targetClass, target.second, data());
  }

  public int size() {
    return items.length();
  }
}
