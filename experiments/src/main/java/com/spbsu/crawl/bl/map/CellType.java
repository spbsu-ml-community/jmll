package com.spbsu.crawl.bl.map;


/**
 * Created by noxoomo on 24/04/16.
 */

public enum CellType {
  WALL(Wall.class),
  FLOOR(Floor.class),
  UNKNOWN(Unknown.class),
  OTHER(Other.class);

  private final Class<?> clazz;

  CellType(Class<?> clazz) {
    this.clazz = clazz;
  }

  public <T> Class<T> clazz() {
    //noinspection unchecked
    return (Class<T>) clazz;
  }

  public static CellType type(int dungeonType) {
    if (4 < dungeonType &&
            dungeonType < 13) {
      return CellType.WALL;
    }
    return CellType.FLOOR;
  }

}

