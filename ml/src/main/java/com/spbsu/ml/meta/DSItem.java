package com.spbsu.ml.meta;

/**
 * User: solar
 * Date: 20.06.14
 * Time: 15:13
 */
public interface DSItem {
  String id();

  abstract class Stub implements DSItem {
    @Override
    public boolean equals(final Object o) {
      if (this == o)
        return true;
      if (!(o instanceof DSItem))
        return false;

      final DSItem other = (DSItem) o;
      return id().equals(other.id());
    }

    @Override
    public int hashCode() {
      return id().hashCode();
    }
  }
}
