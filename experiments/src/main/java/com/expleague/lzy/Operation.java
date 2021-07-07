package com.expleague.lzy;

import com.expleague.lzy.data.DataEntity;

public interface Operation extends Runnable {
  Slot[] input();
  Slot[] output();

  default DataEntity[] entities(Zygote instance) {return new DataEntity[0];}
}
