package com.expleague.zy.data;

import java.util.stream.Stream;

public interface DataSnapshot extends Stream<DataPage> {
  DataStream root();
  long size();

  Comparable version();
}
