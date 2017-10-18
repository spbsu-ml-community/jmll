package com.expleague.crawl.data.impl.system;

import com.expleague.crawl.data.Message;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class RegisterMessage implements Message {
  @JsonProperty
  private String username;
  @JsonProperty
  private String password;
  @JsonProperty
  private String email;

  public RegisterMessage(String username, String password, String email) {
    this.username = username;
    this.password = password;
    this.email = email;
  }

  public RegisterMessage() {
  }
}
