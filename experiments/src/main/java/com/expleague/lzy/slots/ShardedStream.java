package com.expleague.lzy.slots;

import com.expleague.lzy.Slot;
import com.expleague.lzy.data.DataSchema;

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
