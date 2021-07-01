package com.expleague.zy.slots;

import com.expleague.zy.Slot;

public interface KafkaTopic extends Slot {
  @Override
  default Access access() {
    return Access.ReadSeq;
  }

  @Override
  default Type type() {
    return Type.DataPage;
  }
}
