package com.expleague.ml.data.tools;

import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Stream;


import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.func.Factory;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.Pair;
import com.expleague.ml.meta.*;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 12:55
 */
@SuppressWarnings("unchecked")
public class PoolBuilder implements Factory<Pool<? extends DSItem>> {
  private DataSetMeta meta;
  private List<DSItem> items = new ArrayList<>();
  private LinkedHashMap<PoolFeatureMeta, Seq<?>> features = new LinkedHashMap<>();

  @Override
  public Pool<? extends DSItem> create() {
    return create((Class<DSItem>)meta.type());
  }

  public <Item extends DSItem> Pool<Item> create(final Class<Item> clazz) {
    //noinspection SuspiciousToArrayCall
    Seq<Item> ds = new ArraySeq<>(PoolBuilder.this.items.toArray((Item[]) Array.newInstance(PoolBuilder.this.items.get(0).getClass(), PoolBuilder.this.items.size())));
    final Pool<Item> result = new Pool<Item>(meta, ds, features);

    { // verifying lines
      features.forEach((meta, values) -> {
        meta.setOwner(result);
        if (values.length() != items.size())
          throw new RuntimeException("Feature " + meta.toString() + " has " + values.length() + " entries " + " expected " + items.size());
      });
    }

    final Set<String> itemIds = new HashSet<>();
    for (final Item item : (List<Item>) this.items) {
      if (itemIds.contains(item.id()))
        throw new RuntimeException(
            "Contain duplicates! Id = " + item.id()
        );
      itemIds.add(toString());
    }
    meta = null;
    this.items = new ArrayList<>();
    features = new LinkedHashMap<>();
    return result;
  }

  public void setMeta(final DataSetMeta meta) {
    this.meta = meta;
  }

  public void addItem(final DSItem read) {
    items.add(read);
  }

  public void newFeature(final FeatureMeta meta, final Seq<?> values) {
    final JsonFeatureMeta poolMeta = new JsonFeatureMeta(meta, this.meta.id());
    features.put(poolMeta, values);
  }

  public void newTarget(final TargetMeta meta, final Seq<?> target) {
    final JsonTargetMeta poolMeta = new JsonTargetMeta(meta, this.meta.id());
    this.features.put(poolMeta, target);
  }

  public <Item extends DSItem> Stream<Item> items() {
    return (Stream<Item>) items.stream();
  }
}
