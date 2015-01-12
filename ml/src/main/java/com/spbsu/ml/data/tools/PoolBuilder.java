package com.spbsu.ml.data.tools;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


import com.spbsu.commons.func.Factory;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;
import com.spbsu.ml.meta.impl.JsonFeatureMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 12:55
 */
public class PoolBuilder implements Factory<Pool<? extends DSItem>> {
  private JsonDataSetMeta meta;
  private List<DSItem> items = new ArrayList<>();
  private List<Pair<JsonFeatureMeta, Seq<?>>> features = new ArrayList<>();
  private List<Pair<JsonTargetMeta, Seq<?>>> targets = new ArrayList<>();

  @Override
  public Pool<? extends DSItem> create() {
    return create(meta.type().clazz());
  }

  public <Item extends DSItem> Pool<Item> create(final Class<Item> clazz) {
    final Pool<Item> result = new Pool<>(
        meta,
        new ArraySeq<>(items.toArray((Item[])Array.newInstance(items.get(0).getClass(), items.size()))),
        features.toArray((Pair<JsonFeatureMeta, Seq<?>>[]) new Pair[features.size()]),
        targets.toArray((Pair<JsonTargetMeta, Seq<?>>[]) new Pair[targets.size()]));
    { // verifying lines
      for (final Pair<JsonFeatureMeta, Seq<?>> entry : features) {
        entry.getFirst().owner = result;
        if (entry.second.length() != items.size())
          throw new RuntimeException(
              "Feature " + entry.first.toString() + " has " + entry.second.length() + " entries " + " expected " + items.size());
      }
    }
    { // checking targets
      for (final Pair<JsonTargetMeta, Seq<?>> entry : targets) {
        entry.getFirst().owner = result;
        if (entry.second.length() != items.size())
          throw new RuntimeException(
              "Target has " + entry.second.length() + " entries " + " expected " + items.size());
      }
    }

    final Set<String> itemIds = new HashSet<>();
    for (final Item item : (List<Item>)items) {
      if (itemIds.contains(item.id()))
        throw new RuntimeException(
            "Contain duplicates! Id = " + item.id()
        );
      itemIds.add(toString());
    }
    meta = null;
    items = new ArrayList<>();
    features = new ArrayList<>();
    targets = new ArrayList<>();
    return result;
  }

  public void setMeta(final JsonDataSetMeta meta) {
    this.meta = meta;
  }

  public void addItem(final DSItem read) {
    items.add(read);
  }

  public void newFeature(final JsonFeatureMeta meta, final Seq<?> values) {
    features.add(Pair.<JsonFeatureMeta, Seq<?>>create(meta, values));
  }

  public void newTarget(final JsonTargetMeta meta, final Seq<?> target) {
    this.targets.add(Pair.<JsonTargetMeta, Seq<?>>create(meta, target));
  }
}
