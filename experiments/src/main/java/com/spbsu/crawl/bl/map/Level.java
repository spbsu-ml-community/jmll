package com.spbsu.crawl.bl.map;

/**
 * Created by noxoomo on 02/05/16.
 */

public class Level {
  private String identifier;
  private int hashCode;

  public Level(String id) {
    this.identifier = id;
    this.hashCode = id.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    Level level = (Level) o;

    return hashCode == level.hashCode && !(identifier != null ? !identifier.equals(level.identifier) : level.identifier != null);
  }

  @Override
  public int hashCode() {
    return hashCode;
  }
}
