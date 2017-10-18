package com.expleague.sbrealty;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.JsonFeatureMeta;

import java.util.Date;
import java.util.List;
import java.util.function.Consumer;

/**
 * Experts League
 * Created by solar on 10.06.17.
 */
public class FeatureBuilder<T extends DSItem> implements Consumer<T> {
  private final JsonFeatureMeta meta;
  private final VecBuilder builder = new VecBuilder();
  protected Evaluator<T> calc;

  protected FeatureBuilder(String id, String description, Evaluator<T> calc) {
    this.calc = calc;
    this.meta = new JsonFeatureMeta();
    meta.id = id;
    meta.description = description;
    meta.type = FeatureMeta.ValueType.VEC;
  }

  public void init(List<Deal> deals, Vec signal, Date start, Date end) {
    builder.clear();
  }

  public JsonFeatureMeta meta() {
    return meta;
  }

  public Vec build() {
    final Vec build = builder.build();
    builder.clear();
    return build;
  }

  @Override
  public void accept(T t) {
    builder.append(calc.value(t));
  }
}
