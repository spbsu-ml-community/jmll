package com.spbsu.ml.meta.items;

import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 11.07.14
 * Time: 21:25
 */
public class FakeItem implements DSItem {
  public int id;

  public FakeItem() {
  }

  public FakeItem(final int id) {
    this.id = id;
  }

  @Override
  public String id() {
    return "Fake item #" + id;
  }

  @Override
  public String toString() {
    return id();
  }
}
