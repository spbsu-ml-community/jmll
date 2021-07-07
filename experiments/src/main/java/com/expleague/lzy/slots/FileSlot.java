package com.expleague.lzy.slots;

import com.expleague.lzy.Slot;

import java.nio.file.Path;

public interface FileSlot extends Slot {
  Path mount();

  @Override
  default Media media() {
    return Media.File;
  }
}
