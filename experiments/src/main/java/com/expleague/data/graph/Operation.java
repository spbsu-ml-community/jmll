package com.expleague.data.graph;

import com.expleague.data.DataType;

public interface Operation extends Runnable {
  Socket[] sockets();

  DataEntity[] entities(Zygote instance);
  Operation[] path(Operation.Socket from, Operation.Socket to);

  interface Socket {
    default Operation body() {return null;}
    String name();
    AccessType access();
    DataType type();
  }
}
