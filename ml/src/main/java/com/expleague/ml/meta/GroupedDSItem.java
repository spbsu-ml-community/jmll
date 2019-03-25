package com.expleague.ml.meta;

/**
 * User: solar
 * Date: 20.06.14
 * Time: 15:13
 */
public interface GroupedDSItem extends DSItem {
  // what are conditions on group id?
  String groupId();

  abstract class Stub implements GroupedDSItem {

    @Override
    public int hashCode() {
      return this.id().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof GroupedDSItem)) {
        return false;
      }

      final DSItem dsItem = (DSItem) obj;
      return this.id().equals(dsItem.id());
    }
  }
}
