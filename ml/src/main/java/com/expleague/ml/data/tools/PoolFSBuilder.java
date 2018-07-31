package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqBuilder;
import com.expleague.ml.meta.*;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;

import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Stream;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 12:55
 */
@SuppressWarnings("unchecked")
public class PoolFSBuilder<T extends DSItem> implements Pool.Builder<T> {
  private final DataSetMeta meta;
  private final List<T> items = new ArrayList<>();
  private final List<SeqBuilder<?>> builders = new ArrayList<>();

  private final FeatureSet<T> fs;

  public PoolFSBuilder(DataSetMeta meta, FeatureSet<T> fs) {
    this.meta = meta;
    this.fs = fs;
    for (int f = 0; f < fs.dim(); f++) {
      builders.add(fs.meta(f).type().builder());
    }
  }

  public Stream<FeatureSet<? super T>> features() {
    return fs.components();
  }

  public Pool<T> create() {
    final Set<String> itemIds = new HashSet<>();
    for (final T item : this.items) {
      if (itemIds.contains(item.id()))
        throw new RuntimeException("Contain duplicates! Id = " + item.id());
      itemIds.add(toString());
    }

    final LinkedHashMap<PoolFeatureMeta, Seq<?>> features = new LinkedHashMap<>();
    for (int f = 0; f < fs.dim(); f++) {
      final FeatureMeta meta = fs.meta(f);
      if (meta instanceof TargetMeta)
        features.put(new JsonTargetMeta((TargetMeta)meta, this.meta.id()), builders.get(f).build());
      else
        features.put(new JsonFeatureMeta(meta, this.meta.id()), builders.get(f).build());
    }
    //noinspection SuspiciousToArrayCall
    Seq<T> ds = new ArraySeq<>(PoolFSBuilder.this.items.toArray((T[]) Array.newInstance(PoolFSBuilder.this.items.get(0).getClass(), PoolFSBuilder.this.items.size())));
    final Pool<T> result = new Pool<>(meta, ds, features);

    { // verifying lines
      features.forEach((meta, values) -> {
        meta.setOwner(result);
        if (values.length() != items.size())
          throw new RuntimeException("Feature " + meta.toString() + " has " + values.length() + " entries " + " expected " + items.size());
      });
    }

    return result;
  }

  public void accept(final T item) {
    fs.accept(item);
    items.add(item);
  }

  public void advance() {
    final Vec vec = fs.advance();
    for (int f = 0; f < fs.dim(); f++) {
      switch (fs.meta(f).type()) {
        case VEC:
        case SPARSE_VEC:
          ((VecBuilder) builders.get(f)).append(vec.get(f));
          break;
        case INTS:
          ((IntSeqBuilder) builders.get(f)).append((int)vec.get(f));
          break;
        default:
          throw new UnsupportedOperationException("Not implemented yet");
      }
    }
  }
}
