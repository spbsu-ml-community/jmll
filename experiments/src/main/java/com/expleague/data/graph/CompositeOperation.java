package com.expleague.data.graph;

public interface CompositeOperation extends Operation {
  Socket[] allSockets();

  interface Builder {
    Builder append(Operation op);
    Builder link(Operation.Socket from, Operation.Socket to);

    CompositeOperation build();
  }
}
