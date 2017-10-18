package com.expleague.crawl.data.impl.system;

import com.expleague.crawl.data.Message;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class LoginMessage implements Message {
  @JsonProperty
  private String username;
  @JsonProperty
  private String password;

  public LoginMessage(String username, String password) {
    this.username = username;
    this.password = password;
  }

  public LoginMessage() {
  }
}
