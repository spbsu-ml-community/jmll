package com.expleague.zy;

import com.expleague.zy.data.DataEntity;

public interface Operation extends Runnable {
  Slot[] input();
  Slot[] output();

  default DataEntity[] entities(Zygote instance) {return new DataEntity[0];}
}
