package com.expleague.zy;

import java.net.URI;

public interface Zygote {
  Operation operation();
  URI resolve(Slot slot);

  ResourceBindings[] bindings();
  ReproducibilityLevel rl();

  interface ResourceBindings {
    Slot socket();
    URI resource();
  }

  enum ReproducibilityLevel {
    ByteLevel,
    StatLevel,
    SratLevel
  }
}
