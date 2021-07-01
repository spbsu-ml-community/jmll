package com.expleague.zy;

import com.expleague.zy.data.DataEntity;

public interface Operation extends Runnable {
  Slot[] slots();

  DataEntity[] entities(Zygote instance);
}
