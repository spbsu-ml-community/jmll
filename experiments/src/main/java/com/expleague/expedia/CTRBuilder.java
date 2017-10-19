package com.expleague.expedia;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import gnu.trove.map.hash.TObjectIntHashMap;

public class CTRBuilder<T> {
  private final JsonFeatureMeta meta = new JsonFeatureMeta();
  private final VecBuilder ctr = new VecBuilder();
  private final TObjectIntHashMap<T> alpha = new TObjectIntHashMap<>();
  private final TObjectIntHashMap<T> beta = new TObjectIntHashMap<>();

  public CTRBuilder(final String id, final String description) {
    meta.id = id;
    meta.description = description;
    meta.type = FeatureMeta.ValueType.VEC;
  }

  public void add(final T key, final boolean alpha) {
    if (alpha) {
      addAlpha(key);
    } else {
      addBeta(key);
    }
  }

  public void addAlpha(final T key) {
    alpha.adjustOrPutValue(key, 1, 1);
    ctr.append(getCTR(key));
  }

  public void addBeta(final T key) {
    beta.adjustOrPutValue(key, 1, 1);
    ctr.append(getCTR(key));
  }

  public double getCTR(final T key) {
    return (alpha.get(key) + 1.0) / (alpha.get(key) + beta.get(key) + 2.0);
  }

  public Vec build() {
    return ctr.build();
  }

  public JsonFeatureMeta getMeta() {
    return meta;
  }

  // TODO: load builder

  // TODO: save builder
}
