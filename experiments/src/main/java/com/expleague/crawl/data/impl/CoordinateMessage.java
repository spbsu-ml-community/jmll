package com.expleague.crawl.data.impl;

import com.expleague.crawl.data.Message;
import com.fasterxml.jackson.annotation.JsonProperty;

public class CoordinateMessage implements Message {
  @JsonProperty("x")
  private int x;
  @JsonProperty("y")
  private int y;

  public int getY() {
    return y;
  }

  public int getX() {

    return x;
  }
}
