package com.expleague.crawl.data.impl;

/**
 * User: qdeee
 * Date: 04.04.16
 */
public enum KeyCommand {
  PICK_ITEM("g"),
  AUTO_EXPLORE("o"),
  UP_STAIRCASE("<"),
  DOWN_STAIRCASE(">"),
  EAT_FOOD_FROM_FLOOR("e"),


  ;

  private final String text;

  KeyCommand(String text) {
    this.text = text;
  }

  public String getText() {
    return text;
  }
}
