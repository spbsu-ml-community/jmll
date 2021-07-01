package com.expleague.zy.data;

import java.util.List;
import java.util.Map;

public interface DataSchema {
  DataSchema[] parents();
  Map<String, Class> properties();
  List<String> canonicalOrder();

  boolean check(DataPage page);
  boolean check(DataPage.Item item);
}
