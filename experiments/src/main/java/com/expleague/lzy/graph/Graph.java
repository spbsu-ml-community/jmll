package com.expleague.lzy.graph;

import com.expleague.lzy.Slot;
import com.expleague.lzy.Operation;

public interface Graph extends Operation {
  Slot[] allSockets();

  interface Builder {
    Builder append(Operation op);
    Builder link(Slot from, Slot to);

    Graph build();
  }
}
