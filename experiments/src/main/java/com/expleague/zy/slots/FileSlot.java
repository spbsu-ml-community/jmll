package com.expleague.zy.slots;

import com.expleague.zy.Slot;

import java.nio.file.Path;

public interface FileSlot extends Slot {
  Path mount();

  @Override
  default Media media() {
    return Media.File;
  }
}
