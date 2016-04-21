package com.spbsu.crawl.data.impl;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public enum KeyCode {
  UP(-254),
  DOWN(-253),
  LEFT(-252),
  RIGHT(-251),
  a(3),
  n(15),
  q(19),
  EXIT_GAME(19),
  ;

  private final int code;

  KeyCode(int code) {
    this.code = code;
  }

  public int getCode() {
    return code;
  }
}
