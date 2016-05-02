package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class InputCommandMessage implements Message {
  @JsonProperty("text")
  private final String text;

  public InputCommandMessage(KeyCommand keyCommand) {
    this.text = keyCommand.getText();
  }

  public InputCommandMessage(char keyCommand) {
    this.text = Character.toString(keyCommand);
  }

}
