package com.expleague.zy;

import com.expleague.zy.data.DataSchema;

import javax.annotation.Nullable;
import java.net.URI;

public interface Slot {
  URI id();

  Operation body();
  String name();
  Access access();
  Media type();
  @Nullable
  DataSchema data();

  enum Media {
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
    Media(Class type) {

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
