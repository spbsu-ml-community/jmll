package com.expleague.data;

import java.util.stream.Stream;

public enum DataType {
  DataStream(Stream.class),
  PythonType(java.lang.Object.class),

  Properties(java.util.Properties.class),
  String(java.lang.String.class),
  Integer(java.lang.Integer.class),
  File(java.nio.file.Path.class),

  ;

  private final Class type;

  public Class of() {
    return type;
  }
  DataType(Class type) {

    this.type = type;
  }
}
