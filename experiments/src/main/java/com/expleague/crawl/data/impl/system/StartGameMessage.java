package com.expleague.crawl.data.impl.system;

import com.expleague.crawl.data.Message;
import com.fasterxml.jackson.annotation.JsonProperty;

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
