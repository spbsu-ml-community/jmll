package com.expleague.zy.slots;

import com.expleague.zy.Slot;

public interface PythonVar extends Slot {
  String name();

  @Override
  default Access access() {
    return Access.ReadWrite;
  }

  default Media type() {
    return Media.PythonVar;
  }
}
