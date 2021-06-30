package com.expleague.data.graph.resources;

import com.expleague.data.DataType;
import com.expleague.data.graph.AccessType;
import com.expleague.data.graph.Operation;

public class Var implements Operation.Socket {
  private final String name;

  public Var(String name) {
    this.name = name;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public AccessType access() {
    return AccessType.MEM_RW;
  }

  public DataType type() {
    return DataType.PythonType;
  }
}
