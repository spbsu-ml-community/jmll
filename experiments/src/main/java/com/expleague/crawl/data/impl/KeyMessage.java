package com.expleague.crawl.data.impl;

import com.expleague.crawl.data.Message;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class KeyMessage implements Message {
  @JsonProperty("keycode")
  private final int keyCode;

  public KeyMessage(final KeyCode keyCode) {
    this.keyCode = keyCode.getCode();
  }
}
