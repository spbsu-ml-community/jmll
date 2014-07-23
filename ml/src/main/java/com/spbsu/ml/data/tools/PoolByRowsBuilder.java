package com.spbsu.ml.data.tools;

import java.lang.reflect.Array;
import java.util.*;


import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolFeatureMeta;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 12:55
 */
public class PoolByRowsBuilder implements Factory<Pool<? extends DSItem>> {
  private JsonDataSetMeta meta;
  private List<DSItem> items = new ArrayList<>();
  private LinkedHashMap<FeatureMeta, VecBuilder> features = new LinkedHashMap<>();

  @Override
  public Pool<? extends DSItem> create() {
    return create(meta.type().clazz());
  }

  public <Item extends DSItem> Pool<Item> create(Class<Item> clazz) {
    Pair<PoolFeatureMeta, Vec>[] features = new Pair[this.features.size()];
    int index = 0;
    final Holder<DataSet<?>> dataSet = Holder.create(null);
    for (final Map.Entry<FeatureMeta, VecBuilder> entry : this.features.entrySet()) {
      final PoolFeatureMeta key = new PoolFeatureMeta() {
        @Override
        public DataSet<?> associated() {
          return dataSet.getValue();
        }

        @Override
        public String id() {
          return entry.getKey().id();
        }

        @Override
        public String description() {
          return entry.getKey().description();
        }

        @Override
        public ValueType type() {
          return entry.getKey().type();
        }
      };entry.getKey();
      features[index++] = new Pair<>(key, entry.getValue().build());
    }
    final Pool<Item> result = new Pool<>(
        meta,
        new ArraySeq<>(items.toArray((Item[])Array.newInstance(items.get(0).getClass(), items.size()))),
        features,
        new Pair[0]
    );
    { // verifying lines
      dataSet.setValue(result.data());
      for (Pair<PoolFeatureMeta, Vec> entry : features) {
        if (entry.second.length() != items.size())
          throw new RuntimeException(
              "Feature " + entry.first.toString() + " has " + entry.second.length() + " entries " + " expected " + items.size());
      }
    }

    final Set<String> itemIds = new HashSet<>();
    for (Item item : (List<Item>)items) {
      if (itemIds.contains(item.id()))
        throw new RuntimeException(
            "Contain duplicates! Id = " + item.id()
        );
      itemIds.add(toString());
    }
    meta = null;
    items = new ArrayList<>();
    this.features.clear();
    return result;
  }

  public void setMeta(final JsonDataSetMeta meta) {
    this.meta = meta;
  }

  public <T extends DSItem> void addItem(final T item, final Vectorization<T> vec) {
    items.add(item);
    final Vec value = vec.value(item);
    for (int i = 0; i < value.length(); i++) {
      final FeatureMeta fmeta = vec.meta(i);
      VecBuilder builder = features.get(fmeta);
      if (builder == null)
        features.put(fmeta, builder = new VecBuilder());
      builder.add(value.get(i));
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
}
