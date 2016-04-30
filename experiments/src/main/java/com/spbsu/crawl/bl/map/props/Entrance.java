package com.spbsu.crawl.bl.map.props;

/**
 * User: Noxoomo
 * Date: 30.04.16
 * Time: 23:07
 */
public class Entrance implements CellProperty {

  public enum Type {
    Down,
    Up
  }

  private final Type type;

  public Entrance(Type type) {
    this.type = type;
  }

  public Type type() {
    return type;
  }


}
