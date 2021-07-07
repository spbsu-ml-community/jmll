package com.expleague.lzy.slots;

import com.expleague.lzy.Slot;

public interface PythonVar extends Slot {
  String name();

  default Media media() {
    return Media.PythonVar;
  }
}
