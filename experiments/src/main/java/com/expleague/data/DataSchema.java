package com.expleague.data;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface DataSchema {
  DataSchema EMPTY = new EmptyDataSchema();

  DataSchema parent();
  Map<String, Class> properties();
  ReferenceType referenceType(String name);
  List<String> canonicalOrder();

  boolean check(DataPage page);
  boolean check(DataPage.Item item);

  enum ReferenceType {
    Strong,
    Soft,
    Weak
  }

  class EmptyDataSchema implements DataSchema {
    @Override
    public DataSchema parent() {
      return EMPTY;
    }

    @Override
    public Map<String, Class> properties() {
      return Collections.emptyMap();
    }

    @Override
    public ReferenceType referenceType(String name) {
      return null;
    }

    @Override
    public List<String> canonicalOrder() {
      return Collections.emptyList();
    }

    @Override
    public boolean check(DataPage page) {
      return true;
    }

    @Override
    public boolean check(DataPage.Item item) {
      return true;
    }
  }
}
