package com.spbsu.ml.data.tools;

import java.lang.reflect.Array;
import java.util.*;


import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.TargetMeta;
import com.spbsu.ml.meta.impl.JsonPoolMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 12:55
 */
public class PoolBuilder implements Factory<Pool<DSItem>> {
  private JsonPoolMeta meta;
  private List<DSItem> items = new ArrayList<>();
  private List<Pair<FeatureMeta, Vec>> features = new ArrayList<>();
  private Seq<?> target;
  private TargetMeta targetMeta;
  private int lastRegistered = 0;

  @Override
  public Pool<DSItem> create() {
    return create(DSItem.class);
  }

  public <Item extends DSItem> Pool<Item> create(Class<Item> clazz) {
    { // verifying lines
      for (Pair<FeatureMeta, Vec> entry : features) {
        if (entry.second.dim() != items.size())
          throw new RuntimeException(
              "Feature " + entry.first.toString() + " has " + entry.second.dim() + " entries " + " expected " + items.size());
      }
    }
    { // checking targets
      if (target.length() != items.size())
        throw new RuntimeException(
            "Target has " + target.length() + " entries " + " expected " + items.size());
    }

    if (!meta.duplicatesAllowed()) { // check for double items
      final Set<String> itemIds = new HashSet<>();
      for (Item item : (List<Item>)items) {
        if (itemIds.contains(item.id()))
          throw new RuntimeException(
              "Contain duplicates! Id = " + item.id()
          );
        itemIds.add(toString());
      }
    }
    @SuppressWarnings("unchecked")
    final Pool<Item> result = new Pool<>(
        meta,
        new ArraySeq<>(items.toArray((Item[])Array.newInstance(items.get(0).getClass(), items.size()))),
        features.toArray((Pair<FeatureMeta, Vec>[]) new Pair[features.size()]),
        Pair.<TargetMeta, Seq<?>>create(targetMeta, target));
    meta = null;
    items = new ArrayList<>();
    features = new ArrayList<>();
    target = null;
    lastRegistered = 0;
    return result;
  }

  public void setMeta(final JsonPoolMeta meta) {
    this.meta = meta;
  }

  public void addItem(final DSItem read) {
    items.add(read);
  }

  public void newFeature(final FeatureMeta meta, Seq<?> values) {
    features.add(Pair.create(meta, (Vec)values));
  }

  public <T> void newTarget(final TargetMeta meta, Seq<?> target) {
    //noinspection unchecked
    targetMeta = meta;
    this.target = target;
  }
}
