package com.spbsu.crawl.learning.features;

/**
 * Created by noxoomo on 17/07/16.
 */
public class NumericalFeature implements Feature {
  private final int value;
  private final String name;

  public NumericalFeature(final int value, final String name) {
    this.value = value;
    this.name = name;
  }

  @Override
  public int dim() {
    return 1;
  }

  @Override
  public int at(int i) {
    if (i != 0) {
      throw new IllegalArgumentException("Feature dim() == 1");
    }
    return value;
  }

  @Override
  public String name() {
    return name;
  }
}
