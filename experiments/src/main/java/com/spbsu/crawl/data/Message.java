package com.spbsu.crawl.data;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */

@JsonIgnoreProperties(ignoreUnknown = true)
public interface Message {
  default Types type() {
    return Stream.of(Types.values()).filter(t ->
            getClass().equals(t.clazz())
    ).findFirst().orElse(null);
  }
  enum Types {
    PING(PingCommand.class),
    PONG(PongMessage.class),
    LOGIN(LoginMessage.class),
    LOBBY_CLEAR(IgnoreMessage.class),
    LOBBY_COMPLETE(IgnoreMessage.class),
    HTML(IgnoreMessage.class),
    SET_GAME_LINKS(IgnoreMessage.class),
    LOGIN_SUCCESS(LoginSuccessMessage.class),
    ;

    private final Class<?> clazz;

    Types(Class<?> clazz) {
      this.clazz = clazz;
    }

    public <T> Class<T> clazz() {
      //noinspection unchecked
      return (Class<T>)clazz;
    }
  }
}
