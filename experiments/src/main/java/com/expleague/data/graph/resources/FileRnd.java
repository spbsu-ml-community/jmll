package com.expleague.data.graph.resources;

import com.expleague.data.DataType;
import com.expleague.data.graph.AccessType;
import com.expleague.data.graph.Operation;

import java.nio.file.Path;

public class FileRnd implements Operation.Socket {
  private final String name;
  private final Path mount;

  public FileRnd(String name, Path mount) {
    this.name = name;
    this.mount = mount;
  }

  public Path mount() {
    return mount;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public AccessType access() {
    return AccessType.RANDOM_READ;
  }

  @Override
  public DataType type() {
    return DataType.File;
  }
}
