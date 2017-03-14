package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

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
