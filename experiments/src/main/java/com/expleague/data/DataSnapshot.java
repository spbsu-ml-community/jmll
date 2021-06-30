package com.expleague.data;

import java.util.UUID;
import java.util.stream.Stream;

public interface DataSnapshot extends Stream<DataPage> {
  DataStream root();
  UUID[] pages();
  long size();
}
