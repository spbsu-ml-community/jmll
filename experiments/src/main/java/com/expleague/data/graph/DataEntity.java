package com.expleague.data.graph;

import com.expleague.data.DataPage;
import com.expleague.data.DataSchema;

import java.net.URI;

public interface DataEntity extends DataPage {
  Zygote source();
  Component[] components();

  interface Component {
    URI id();
    Operation.Socket source();
    DataSchema.ReferenceType referenceType();
  }
}
