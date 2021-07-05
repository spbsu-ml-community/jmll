package com.expleague.zy.slots;

import com.expleague.zy.Slot;

public interface PythonVar extends Slot {
  String name();

  default Media media() {
    return Media.PythonVar;
  }
}
