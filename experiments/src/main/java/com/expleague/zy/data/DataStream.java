package com.expleague.zy.data;

import java.util.stream.Stream;

public interface DataStream extends Stream<DataPage> {
  DataSnapshot snapshot();

  Comparable version();
}
