package com.spbsu.crawl.learning.features;

import com.spbsu.crawl.bl.helpers.CategoricalFeaturesMap;
import gnu.trove.set.TIntSet;

/**
 * Created by noxoomo on 17/07/16.
 */
public class CategoricalFeature implements Feature {
  private final CategoricalFeaturesMap index;
  private final TIntSet values;
  private final String name;

  public CategoricalFeature(final CategoricalFeaturesMap index,
                            final TIntSet values,
                            final String name) {
    this.index = index;
    this.values = values;
    this.name = name;
  }

  @Override
  public int dim() {
    return index.dictSize();
  }

  @Override
  public int at(int i) {
    if (i >= dim()) {
      throw new IllegalArgumentException("Error: index >= dim()");
    }
    return values.contains(i) ? 1 : 0;
  }

  @Override
  public String name() {
    return name;
  }
}
