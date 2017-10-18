package com.expleague.crawl.data;

import com.expleague.crawl.data.impl.*;
import com.expleague.crawl.data.impl.system.*;
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
  REGISTER(RegisterMessage.class),
  LOBBY_ENTRY(IgnoreMessage.class),
  LOBBY_REMOVE(IgnoreMessage.class),
  LOBBY_CLEAR(IgnoreMessage.class),
  MENU(MenuMessage.class),
  LOBBY_COMPLETE(LobbyComplete.class),
  LOGIN_SUCCESS(LoginSuccessMessage.class),
  GO_LOBBY(GoLobbyMessage.class),
  PLAY(StartGameMessage.class),
  WATCH(StartSpectateMessage.class),
  HTML(IgnoreMessage.class),
  SET_GAME_LINKS(SetGameLinks.class),
  UPDATE_SPECTATORS(IgnoreMessage.class),
  UI_STATE(IgnoreMessage.class),
  OPTIONS(IgnoreMessage.class),
  GAME_CLIENT(IgnoreMessage.class),
  GAME_STARTED(GameStarted.class),
  GAME_ENDED(GameEnded.class),
  VERSION(IgnoreMessage.class),
  LAYOUT(IgnoreMessage.class),
  INPUT_MODE(InputModeMessage.class),
  CURSOR(IgnoreMessage.class),
  MSGS(IgnoreMessage.class),
  PLAYER(PlayerInfoMessage.class),
  MAP(UpdateMapMessage.class),
  MON(MonsterInfoMessage.class),
  KEY(KeyMessage.class),
  INPUT(InputCommandMessage.class),
  CLOSE(IgnoreMessage.class),
  TXT(IgnoreMessage.class),
  STALE_PROCESSES(IgnoreMessage.class),
  DELAY(IgnoreMessage.class),
  CLEAR_OVERLAYS(IgnoreMessage.class),
  OVERLAY(IgnoreMessage.class),
  UPDATE_MENU(IgnoreMessage.class),
  CLOSE_MENU(IgnoreMessage.class),
  FLASH(IgnoreMessage.class),
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
