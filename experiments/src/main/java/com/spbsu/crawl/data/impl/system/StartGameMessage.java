package com.spbsu.crawl.data.impl.system;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class StartGameMessage implements Message {
  @JsonProperty("game_id")
  private final String gameId;

  public StartGameMessage(String gameId) {
    this.gameId = gameId;
  }
}
