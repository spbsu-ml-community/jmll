package com.spbsu.crawl.bl.map;

import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.Optional;

/**
 * Created by noxoomo on 02/05/16.
 */

public class Layer {
  private final TIntObjectHashMap<TerrainType> terrains;
  private final Level level;

  public Layer(Level level) {
    this.terrains = new TIntObjectHashMap<>();
    this.level = level;
  }

  private int id(int x, int y) {
    return x + 100000 * y;
  }

  public Level level() {
    return level;
  }

  void clear() {
    terrains.clear();
  }

  Optional<TerrainType> getCell(final int x, final int y) {
    final int key = id(x, y);
    if (terrains.containsKey(key)) {
      return Optional.of(terrains.get(key));
    } else {
      return Optional.empty();
    }
  }

  void setCell(final int x, final int y,
               final TerrainType type) {
    final int key = id(x, y);
    terrains.put(key, type);
  }
}
