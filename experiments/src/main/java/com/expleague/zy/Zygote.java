package com.expleague.zy;

import java.net.URI;

public interface Zygote {
  Operation operation();
  URI resolve(Slot slot);

  ResourceBindings[] bindings();

  interface ResourceBindings {
    Slot socket();
    URI resource();
  }
}
