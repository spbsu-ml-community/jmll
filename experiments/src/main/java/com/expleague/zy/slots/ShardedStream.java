package com.expleague.zy.slots;

import com.expleague.zy.Slot;
import com.expleague.zy.data.DataSchema;

import javax.annotation.Nonnull;

public interface ShardedStream extends Slot {
  String keyName();

  @Nonnull
  DataSchema contentType();

  @Override
  default Media media() {
    return Media.Pipe;
  }
}
