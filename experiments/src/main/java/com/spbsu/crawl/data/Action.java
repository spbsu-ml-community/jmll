package com.spbsu.crawl.data;

/**
 * User: qdeee
 * Date: 05.05.16
 */
public enum Action {
  MOVE_UP('k'),
  MOVE_DOWN('j'),
  MOVE_LEFT('h'),
  MOVE_RIGHT('l'),
  MOVE_UP_RIGHT('u'),
  MOVE_UP_LEFT('y'),
  MOVE_DOWN_RIGHT('n'),
  MOVE_DOWN_LEFT('b'),;


  private char code;

  Action(char code) {
    this.code = code;
  }

  public char code() {
    return code;
  }
}
