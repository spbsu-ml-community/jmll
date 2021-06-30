package com.expleague.data.graph.resources;

import com.expleague.data.DataSchema;
import com.expleague.data.DataType;
import com.expleague.data.graph.AccessType;
import com.expleague.data.graph.Operation;

public class KafkaTopic implements Operation.Socket {
  private final String name;
  private final DataSchema schema;

  public KafkaTopic(String name, DataSchema schema) {
    this.name = name;
    this.schema = schema;
  }

  @Override
  public String name() {
    return name;
  }

  public DataSchema schema() {
    return schema;
  }

  @Override
  public AccessType access() {
    return AccessType.SEQ_READ;
  }

  @Override
  public DataType type() {
    return DataType.DataStream;
  }
}
