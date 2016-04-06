package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

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
