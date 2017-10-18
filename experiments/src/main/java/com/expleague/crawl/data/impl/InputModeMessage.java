package com.expleague.crawl.data.impl;

import com.expleague.crawl.data.Message;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * User: qdeee
 * Date: 04.04.16
 */
public class InputModeMessage implements Message {
  public static final int START_INPUT = 0;
  public static final int END_INPUT = 1;

  @JsonProperty("mode")
  private int inputMode;

  public int inputMode() {
    return inputMode;
  }

  public static boolean isStartMessage(final Message message) {
    if (message instanceof InputModeMessage) {
      final InputModeMessage inputModeMessage = (InputModeMessage) message;
      return inputModeMessage.inputMode() == InputModeMessage.START_INPUT;
    }
    return false;
  }

  public static boolean isEndMessage(final Message message) {
    if (message instanceof InputModeMessage) {
      final InputModeMessage inputModeMessage = (InputModeMessage) message;
      return inputModeMessage.inputMode() == InputModeMessage.END_INPUT;
    }
    return false;
  }
}
