package com.expleague.zy;

import java.net.URI;

public interface Slot {
  URI id();

  Operation body();
  String name();
  Access access();
  Type type();

  enum Type {
    DataPage(com.expleague.zy.data.DataPage.class),
    PythonVar(Object.class),

    Properties(java.util.Properties.class),
    String(java.lang.String.class),
    Integer(java.lang.Integer.class),
    File(java.nio.file.Path.class),

    ;

    private final Class type;

    public Class of() {
      return type;
    }
    Type(Class type) {

      this.type = type;
    }
  }

  enum Access {
    ReadSeq,
    ReadAcc,
    ReadWrite,
    WriteAcc,
    WriteSeq
  }
}
