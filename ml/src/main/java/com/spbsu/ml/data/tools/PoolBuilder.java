package com.spbsu.ml.data.tools;

import java.lang.reflect.Array;
import java.util.*;


import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.TargetMeta;
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
  private List<Pair<JsonFeatureMeta, Vec>> features = new ArrayList<>();
  private Seq<?> target;
  private JsonTargetMeta targetMeta;

  @Override
  public Pool<? extends DSItem> create() {
    return create(meta.type().clazz());
  }

  public <Item extends DSItem> Pool<Item> create(Class<Item> clazz) {
    final Pool<Item> result = new Pool<>(
        meta,
        new ArraySeq<>(items.toArray((Item[])Array.newInstance(items.get(0).getClass(), items.size()))),
        features.toArray((Pair<JsonFeatureMeta, Vec>[]) new Pair[features.size()]),
        Pair.<TargetMeta, Seq<?>>create(targetMeta, target));
    { // verifying lines
      for (Pair<JsonFeatureMeta, Vec> entry : features) {
        entry.getFirst().owner = result;
        if (entry.second.dim() != items.size())
          throw new RuntimeException(
              "Feature " + entry.first.toString() + " has " + entry.second.dim() + " entries " + " expected " + items.size());
      }
    }
    { // checking targets
      targetMeta.owner = result;
      if (target.length() != items.size())
        throw new RuntimeException(
            "Target has " + target.length() + " entries " + " expected " + items.size());
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
    features = new ArrayList<>();
    target = null;
    return result;
  }

  public void setMeta(final JsonDataSetMeta meta) {
    this.meta = meta;
  }

  public void addItem(final DSItem read) {
    items.add(read);
  }

  public void newFeature(final JsonFeatureMeta meta, Seq<?> values) {
    features.add(Pair.create(meta, (Vec)values));
  }

  public <T> void newTarget(final JsonTargetMeta meta, Seq<?> target) {
    //noinspection unchecked
    targetMeta = meta;
    this.target = target;
  }
}
