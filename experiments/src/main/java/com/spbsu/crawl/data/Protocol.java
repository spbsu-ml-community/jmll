package com.spbsu.crawl.data;

import com.spbsu.crawl.data.impl.*;
import com.spbsu.crawl.data.impl.system.*;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public enum Protocol {
  PING(PingCommand.class),
  PONG(PongMessage.class),
  LOGIN(LoginMessage.class),
  LOBBY_ENTRY(IgnoreMessage.class),
  LOBBY_REMOVE(IgnoreMessage.class),
  LOBBY_CLEAR(IgnoreMessage.class),
  LOBBY_COMPLETE(IgnoreMessage.class),
  LOGIN_SUCCESS(LoginSuccessMessage.class),
  GO_LOBBY(GoLobbyMessage.class),
  PLAY(StartGameMessage.class),
  WATCH(StartSpectateMessage.class),
  HTML(IgnoreMessage.class),
  SET_GAME_LINKS(IgnoreMessage.class),
  UPDATE_SPECTATORS(IgnoreMessage.class),
  UI_STATE(IgnoreMessage.class),
  OPTIONS(IgnoreMessage.class),
  GAME_CLIENT(IgnoreMessage.class),
  GAME_STARTED(IgnoreMessage.class),
  VERSION(IgnoreMessage.class),
  LAYOUT(IgnoreMessage.class),
  INPUT_MODE(InputModeMessage.class),
  CURSOR(IgnoreMessage.class),
  MSGS(IgnoreMessage.class),
  PLAYER(PlayerInfoMessage.class),
  MAP(MapInfoMessage.class),
  KEY(KeyMessage.class),
  INPUT(InputCommandMessage.class),
  ;

  private final Class<?> clazz;

  Protocol(Class<?> clazz) {
    this.clazz = clazz;
  }

  public <T> Class<T> clazz() {
    //noinspection unchecked
    return (Class<T>) clazz;
  }
}
