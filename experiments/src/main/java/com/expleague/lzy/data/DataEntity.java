package com.expleague.lzy.data;

import com.expleague.lzy.Slot;
import com.expleague.lzy.Zygote;

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
