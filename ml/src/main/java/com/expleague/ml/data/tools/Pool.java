package com.expleague.ml.data.tools;

import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.expleague.commons.math.vectors.impl.mx.ColsVecSeqMx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.VecSeq;
import com.expleague.commons.system.RuntimeUtils;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.Vectorization;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.meta.*;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TObjectIntHashMap;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.lang.reflect.Array;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 20:55
 *
 */
public class Pool<I extends DSItem> {
  protected final DataSetMeta meta;
  protected final Seq<I> items;
  protected final PoolFeatureMeta[] featuresMeta;
  protected final Seq<?>[] featuresValues;
  protected final List<PoolTargetMeta> targetsMeta;
  protected final List<Seq<?>> targetsValues;

  protected DataSet<I> data;
  protected VecDataSet vecDataSet;

  public interface Builder<T extends DSItem> extends Factory<Pool<T>>, Consumer<T> {
    Pool<T> create();

    void accept(final T item);
    void advance();

    Stream<FeatureSet<? super T>> features();
  }

  @SuppressWarnings("unchecked")
  public static <I extends DSItem> Builder<I> builder(DataSetMeta meta, FeatureSet<? super I>... features) {
    return new PoolFSBuilder<>(meta, FeatureSet.join(features));
  }

  public Pool(final DataSetMeta meta, final Seq<I> items, final LinkedHashMap<PoolFeatureMeta, Seq<?>> features) {
    this.meta = meta;
    this.items = items;
    final List<PoolFeatureMeta> featuresMetas = new ArrayList<>();
    final List<Seq<?>> featuresValues = new ArrayList<>();
    final List<PoolTargetMeta> targetsMetas = new ArrayList<>();
    final List<Seq<?>> targetsValues = new ArrayList<>();
    features.forEach((fmeta, fvalues) -> {
      if (fvalues.length() != items.length())
        throw new IllegalArgumentException("Feature " + fmeta.toString() + " has " + fvalues.length() + " entries expected " + items.length());
      if (fmeta instanceof PoolTargetMeta) {
        targetsMetas.add((PoolTargetMeta) fmeta);
        targetsValues.add(fvalues);
      }
      else {
        featuresMetas.add(fmeta);
        featuresValues.add(fvalues);
      }
      fmeta.setOwner(this);
    });
    meta.setOwner(this);
    this.featuresMeta = featuresMetas.toArray(new PoolFeatureMeta[featuresMetas.size()]);
    //noinspection SuspiciousToArrayCall
    this.featuresValues = (Seq<?>[])featuresValues.toArray(new Seq[featuresValues.size()]);
    this.targetsMeta = targetsMetas;
    //noinspection SuspiciousToArrayCall
    this.targetsValues = targetsValues;
  }

  public DataSetMeta meta() {
    return meta;
  }

  public synchronized DataSet<I> data() {
    if (data == null) {
      final TObjectIntHashMap<I> indices = new TObjectIntHashMap<>(items.length() * 2, 0.7f);
      for (int i = 0; i < items.length(); i++) {
        indices.put(items.at(i), i);
      }
      data = new DataSet.Stub<I>(null) {
        @Override
        public I at(final int i) {
          return items.at(i);
        }

        @Override
        public int length() {
          return items.length();
        }

        @Override
        public DataSetMeta meta() {
          return meta;
        }

        @Override
        public int index(final I obj) {
          return indices.get(obj);
        }

        @Override
        public Class<? extends I> elementType() {
          return items.elementType();
        }
      };
    }
    return data;
  }

  public <T extends DSItem> VecDataSet joinFeatures(final int[] indices, final DataSet<T> ds) {
    final List<Seq<?>> cols = new ArrayList<>();
    boolean hasVecFeatures = false;
    for (int i = 0; i < indices.length; i++) {
      final Seq<?> val = featuresValues[indices[i]];
      cols.add(val);
      if (!hasVecFeatures && (val instanceof VecSeq || val instanceof ArraySeq)) {
        hasVecFeatures = true;
      }
    }

    final Mx data;
    final int[] cumulativeFeatureLengths;
    if (hasVecFeatures) {
      final List<Seq<?>> seqs = new ArrayList<>(cols.size());
      cumulativeFeatureLengths = new int[cols.size()];

      for (int i = 0; i < cols.size(); i++) {
        final Seq<?> col = cols.get(i);
        final int prevFeaturesLength = i > 0 ? cumulativeFeatureLengths[i - 1] : 0;

        if (col instanceof Vec) {
          seqs.add(new VecSeq((Vec) col));
          cumulativeFeatureLengths[i] = prevFeaturesLength + 1;

        } else if (col instanceof VecSeq) {
          seqs.add(col);
          cumulativeFeatureLengths[i] = prevFeaturesLength + col.length();

        } else if (col instanceof ArraySeq) {
          //noinspection unchecked
          seqs.add(new VecSeq((ArraySeq) col));
          cumulativeFeatureLengths[i] = prevFeaturesLength + col.length();

        } else {
          throw new IllegalArgumentException("unexpected feature type " + col.getClass().getSimpleName());
        }
      }

      //noinspection SuspiciousToArrayCall
      data = new ColsVecSeqMx(seqs.toArray(new VecSeq[seqs.size()]));
    } else {
      //noinspection SuspiciousToArrayCall
      data = new ColsVecArrayMx(cols.toArray(new Vec[cols.size()]));
      cumulativeFeatureLengths = ArrayTools.sequence(0, cols.size());
    }

    return new VecDataSetImpl(ds, data, new Vectorization<T>() {
      @Override
      public Vec value(final T subject) {
        return data.row(ds.index(subject));
      }

      @Override
      public FeatureMeta meta(final int findex) {
        final int search = Arrays.binarySearch(cumulativeFeatureLengths, findex);
        final int sourceFeatureIdx = search >= 0 ? search : -search - 1;
        return featuresMeta[indices[sourceFeatureIdx]];
      }

      @Override
      public int dim() {
        return indices.length;
      }
    });
  }

  public VecDataSet vecData() {
    if (vecDataSet == null) {
      final Class[] supportedFeatureTypes = new Class[]{Vec.class, VecSeq.class};
      final DataSet<I> ds = data();
      final TIntArrayList toJoin = new TIntArrayList(featuresValues.length);
      for (int i = 0; i < featuresValues.length; i++) {
        for (final Class clazz : supportedFeatureTypes) {
          //noinspection unchecked
          if (clazz.isAssignableFrom(featuresMeta[i].type().clazz())) {
            toJoin.add(i);
            break;
          }
        }
      }
      vecDataSet = joinFeatures(toJoin.toArray(), ds);
    }
    return vecDataSet;
  }

  public synchronized void addTarget(final TargetMeta meta, final Seq<?> target) {
    JsonTargetMeta targetMeta = new JsonTargetMeta(meta, data().meta().id());
    targetsMeta.add(targetMeta);
    targetMeta.setOwner(this);
    targetsValues.add(target);
  }

  public Pool<I> sub(int[] indices) {
    final JsonDataSetMeta meta = new JsonDataSetMeta(this.meta.source(), this.meta.author(), new Date(), this.meta.type(), this.meta.id() + "-sub-" + ArrayTools.sum(indices));
    final LinkedHashMap<PoolFeatureMeta, Seq<?>> features = new LinkedHashMap<>();
    for (int f = 0; f < featuresMeta.length; f++) {
      features.put(new JsonFeatureMeta(featuresMeta[f], meta.id()), featuresValues[f].sub(indices));
    }
    for (int t = 0; t < targetsMeta.size(); t++) {
      features.put(new JsonTargetMeta(targetsMeta.get(t), meta.id()), targetsValues.get(t).sub(indices));
    }
    return new Pool<>(meta, items.sub(indices), features);
  }

  public <T extends TargetFunc> T target(final Class<T> targetClass) {
    for (int i = targetsValues.size() - 1; i >= 0; i--) {
      final T target = RuntimeUtils.newInstanceByAssignable(targetClass, targetsValues.get(i), targetsMeta.get(i).associated());
      if (target != null)
        return target;
    }
    try {
      return multiTarget(targetClass);
    } catch (Exception e) {
      throw new RuntimeException("No proper constructor found", e);
    }
  }

  public Seq<?> target(String name) {
    for (int i = targetsValues.size() - 1; i >= 0; i--) {
      if (targetsMeta.get(i).id().equals(name))
        return targetsValues.get(i);
    }
    throw new RuntimeException("No such target: " + name);
  }

  public Seq<?> target(int index) {
    return targetsValues.get(index);
  }

  public <T extends TargetFunc> T multiTarget(final Class<T> targetClass) {
    final Mx targetsValues = new VecBasedMx(size(), targetsMeta.size());
    for (int j = 0; j < this.targetsValues.size(); j++) {
      final Seq<?> target = this.targetsValues.get(j);
      if (target instanceof Vec) {
        VecTools.assign(targetsValues.col(j), (Vec) target);
      }
      else if (target instanceof IntSeq) {
        final IntSeq intSeq = (IntSeq) target;
        for (int i = 0; i < target.length(); i++)
          targetsValues.set(i, j, intSeq.intAt(i));
      }
      else {
        throw new RuntimeException("Unsupported target type: " + target.getClass().getName());
      }
    }

    final T target = RuntimeUtils.newInstanceByAssignable(targetClass, targetsValues, data());
    if (target != null) {
      return target;
    } else {
      throw new RuntimeException("No proper constructor found");
    }
  }

  public int size() {
    return items.length();
  }

  public DataSet<?> data(final String dsId) {
    final DataSet<I> data = data();
    if (data.meta().id().equals(dsId))
      return data;
    throw new IllegalArgumentException("No such dataset in the pool");
  }


  @Override
  public boolean equals(final Object obj) {
    if (obj == this)
      return true;
    if (obj == null || obj.getClass() != getClass())
      return false;

    final Pool other = (Pool) obj;
    return new EqualsBuilder().append(meta, other.meta).
        append(items, other.items).
        append(featuresMeta, other.featuresMeta).
        append(targetsMeta, other.targetsMeta).
        isEquals();
  }

  @Override
  public int hashCode() {
    return new HashCodeBuilder().append(meta).
        append(items).
        append(featuresMeta).
        append(targetsMeta).
        toHashCode();
  }

  public PoolFeatureMeta[] features() {
    return featuresMeta;
  }

  public <T extends TargetFunc> T target(String name, Class<T> targetClass) {
    for (int i = targetsMeta.size() - 1; i >= 0; i--) {
      final PoolTargetMeta current = targetsMeta.get(i);
      if (!current.id().equals(name))
        continue;
      final T target = RuntimeUtils.newInstanceByAssignable(targetClass, targetsValues.get(i), current.associated());
      if (target != null)
        return target;
    }

    throw new RuntimeException("No such target: " + name + " of type " + targetClass.getSimpleName());
  }

  public <T> T feature(int findex, int iindex) {
    //noinspection unchecked
    return (T)featuresValues[findex].at(iindex);
  }

  public TargetFunc targetByName(String metricName) {
    final String loss;
    String name = null;
    if (metricName.contains("(")) { // has taget name
      final int nameIndex = metricName.indexOf("(") + 1;
      loss = metricName.substring(0, nameIndex);
      name = metricName.substring(nameIndex + 1, metricName.length() - 1);
    }
    else loss = metricName;

    final Class<? extends TargetFunc> lossFunc = DataTools.targetByName(loss);

    return name != null ? target(name, lossFunc) : target(lossFunc);
  }

  public <T> Seq<T> fdata(int i) {
    //noinspection unchecked
    return (Seq<T>) featuresValues[i];
  }

  public PoolFeatureMeta fmeta(int i) {
    return featuresMeta[i];
  }

  public int fcount() {
    return featuresMeta.length;
  }

  public <T> Seq<T> tdata(int i) {
    //noinspection unchecked
    return (Seq<T>) targetsValues.get(i);
  }

  public PoolFeatureMeta tmeta(int i) {
    return targetsMeta.get(i);
  }

  public int tcount() {
    return targetsMeta.size();
  }
}
