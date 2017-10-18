package com.expleague.crawl.data.impl.system;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.expleague.crawl.data.Message;

/**
 * User: qdeee
 * Date: 07.04.16
 */
public class StartSpectateMessage implements Message {
  @JsonProperty("username")
  private final String username;

  public StartSpectateMessage(final String username) {
    this.username = username;
  }
}
