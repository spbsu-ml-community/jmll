package com.expleague.zy.graph;

import com.expleague.zy.Slot;
import com.expleague.zy.Operation;

public interface Graph extends Operation {
  Slot[] allSockets();

  interface Builder {
    Builder append(Operation op);
    Builder link(Slot from, Slot to);

    Graph build();
  }
}
