package com.spbsu.ml.data.tools;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVecBuilder;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.seq.*;
import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.meta.*;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;
import com.spbsu.ml.meta.impl.PoolFeatureMetaImpl;
import com.spbsu.ml.meta.impl.TargetFeatureImpl;
import com.spbsu.ml.meta.impl.fake.FakeFeatureMeta;
import com.spbsu.ml.meta.impl.fake.FakeTargetMeta;
import com.spbsu.ml.meta.items.FakeItem;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.lang.reflect.Array;
import java.util.*;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 12:55
 */
@SuppressWarnings("UnusedDeclaration")
public class PoolByRowsBuilder<Item extends DSItem> implements Factory<Pool<Item>> {
  private JsonDataSetMeta meta = new JsonDataSetMeta();
  private List<Item> items = new ArrayList<>();

  public PoolByRowsBuilder() {
    this(DataSetMeta.ItemType.FAKE);
  }

  public PoolByRowsBuilder(DataSetMeta.ItemType type) {
    meta.type = type;
    meta.author = System.getProperty("user.name");
    meta.created = new Date();
    StackTraceElement[] stack = Thread.currentThread ().getStackTrace ();
    StackTraceElement main = stack[stack.length - 1];
    meta.source = main.getClassName();
    meta.id = "Unknown pool";
  }

  @Override
  public Pool<Item> create() {
    //noinspection unchecked
    return create((Class<Item>)meta.type().clazz());
  }

  public Pool<Item> create(final Class<Item> clazz) {
    @SuppressWarnings("unchecked")
    final Pair<PoolFeatureMeta, Seq<?>>[] features = new Pair[this.featureMetas.size()];
    @SuppressWarnings("unchecked")
    final Pair<TargetMeta, Seq<?>>[] targets = new Pair[0];
    @SuppressWarnings({"unchecked", "SuspiciousToArrayCall"})
    final Item[] items = this.items.toArray((Item[])Array.newInstance(clazz, this.items.size()));
    final Pool<Item> result = new Pool<>(meta, new ArraySeq<>(items), features, targets);
    meta.owner = result;
    final DataSet<Item> ds = result.data();
    final Holder<DataSet<?>> dataSet = Holder.create(null);
    for (int i = 0; i < featureMetas.size(); i++) {
      final FeatureMeta meta = featureMetas.get(i);
      features[i] = Pair.<PoolFeatureMeta, Seq<?>>create(
              new PoolFeatureMetaImpl(ds, meta),
              featureBuilders.get(i).build()
      );
      featureBuilders.set(i, createBuilderByMeta(meta)); // cleanup
    }
    for (int i = 0; i < targetMetas.size(); i++) {
      final TargetMeta meta = targetMetas.get(i);
      result.addTarget(
              new TargetFeatureImpl(ds, meta),
              targetBuilders.get(i).build()
      );
      targetBuilders.set(i, createBuilderByMeta(meta)); // cleanup
    }
    { // verifying lines
      dataSet.setValue(result.data());
      for (final Pair<PoolFeatureMeta, Seq<?>> entry : features) {
        if (entry.second.length() != this.items.size())
          throw new RuntimeException(
              "Feature " + entry.first.toString() + " has " + entry.second.length() + " entries " + " expected " + this.items.size());
      }
    }

    final Set<String> itemIds = new HashSet<>();
    for (final Item item : this.items) {
      if (itemIds.contains(item.id()))
        throw new RuntimeException(
            "Contain duplicates! Id = " + item.id()
        );
      itemIds.add(toString());
    }
    this.items.clear();
    return result;
  }

  public void setMeta(final JsonDataSetMeta meta) {
    this.meta = meta;
  }

  public void setItemType(DataSetMeta.ItemType type) {
    this.meta.type = type;
  }

  public void addItem(final Item item, final Vectorization<Item> vec) {
    items.add(item);
    if (featureBuilders.size() == 0) { //
      for (int i = 0; i < vec.dim(); i++)
        addFeature(vec.meta(i));
    }
    final Vec value = vec.value(item);
    for (int i = 0; i < value.length(); i++) {
      //noinspection unchecked
      ((SeqBuilder<Object>)featureBuilders.get(index(vec.meta(i)))).add(value.get(i));
    }
  }

  public int size() {
    return items.size();
  }

  public String crc() {
    int crc = 0;
    for (int i = 0; i < items.size(); i++) {
      crc <<= 1;
      crc += items.get(i).hashCode();
    }
    return "" + crc;
  }

  private final List<FeatureMeta> featureMetas = new ArrayList<>();
  private final List<SeqBuilder<?>> featureBuilders = new ArrayList<>();

  private final List<TargetMeta> targetMetas = new ArrayList<>();
  private final List<SeqBuilder<?>> targetBuilders = new ArrayList<>();

  private final TObjectIntHashMap<FeatureMeta> index = new TObjectIntHashMap<>();

  public int allocateFakeFeatures(int count, FeatureMeta.ValueType type) {
    int result = featureMetas.size() - 1;
    for (int i = 0; i < count; i++) {
      final FakeFeatureMeta fmeta = new FakeFeatureMeta(result + 1, type);
      result = addFeature(fmeta);
    }
    return result;
  }

  public int allocateFakeTarget(FeatureMeta.ValueType type) {
    return addTarget(new FakeTargetMeta(type, targetMetas.size()));
  }

  public int index(FeatureMeta meta) {
    return index.get(meta);
  }

  public int addTarget(TargetMeta meta) {
    final int id = targetMetas.size();
    targetMetas.add(meta);
    targetBuilders.add(createBuilderByMeta(meta));
    targetsSet.set(id);
    index.put(meta, id);
    return id;
  }

  public int addFeature(FeatureMeta meta) {
    final int id = featureMetas.size();
    featureMetas.add(meta);
    featureBuilders.add(createBuilderByMeta(meta));
    featuresSet.set(id);
    index.put(meta, id);
    return id;
  }

  private SeqBuilder<?> createBuilderByMeta(FeatureMeta fmeta) {
    SeqBuilder builder;
    switch (fmeta.type()) {
      case VEC:
        builder = new VecBuilder();
        break;
      case SPARSE_VEC:
        builder = new SparseVecBuilder();
        break;
      case INTS:
        builder = new IntSeqBuilder();
        break;
      case CHAR_SEQ:
        builder = new ArraySeqBuilder<>(CharSeq.class);
        break;
      default:
        builder = new ArraySeqBuilder<>(fmeta.type().clazz());
    }
    return builder;
  }

  private final BitSet featuresSet = new BitSet();
  private final BitSet targetsSet = new BitSet();

  public void setFeatures(int offset, Seq next) {
    for (int i = 0; i < next.length(); i++)
      setFeature(offset + i, next.at(i));
  }

  public <T> void setFeature(int offset, T next) {
    if (!featuresSet.get(offset))
      throw new IllegalArgumentException("Feature is already set");
    //noinspection unchecked
    ((SeqBuilder<T>)featureBuilders.get(offset)).add(next);
    featuresSet.clear(offset);
  }

  public <T> void setTarget(int offset, T next) {
    if (!targetsSet.get(offset))
      throw new IllegalArgumentException("Target is already set");
    //noinspection unchecked
    ((SeqBuilder<Object>) targetBuilders.get(offset)).add(next);
    targetsSet.clear(offset);
  }

  public void nextItem() {
    //noinspection unchecked
    nextItem((Item)new FakeItem(items.size()));
  }

  public void nextItem(Item item) {
    if (!featuresSet.isEmpty())
      throw new RuntimeException("Not all features are set for item " + item + " " + featuresSet);
    if (!targetsSet.isEmpty())
      throw new RuntimeException("Not all targets are set for item " + item + " " + targetsSet);
    featuresSet.set(0, featureMetas.size());
    targetsSet.set(0, targetMetas.size());
    items.add(item);
  }
}
