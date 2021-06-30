package com.expleague.data.graph;

import java.net.URI;

public interface Zygote {
  Operation operation();
  ResourceBindings[] bindings();

  interface ResourceBindings {
    Operation.Socket socket();
    URI resource();
  }
}
