package com.expleague.zy.data;

import com.expleague.zy.Slot;
import com.expleague.zy.Zygote;

import java.net.URI;

public interface DataEntity extends DataPage {
  Zygote source();
  Component[] components();

  interface Component {
    URI id();
    Slot source();
    Necessity necessity();

    enum Necessity {
      Needed,
      Supplementary,
      GoodToKnow,
      Temp
    }
  }
}
