package com.spbsu.crawl.bl.map;

import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.Optional;

/**
 * Created by noxoomo on 02/05/16.
 */

public class Layer {
  private final TIntObjectHashMap<TerrainType> terrains = new TIntObjectHashMap<>();

  private int id(int x, int y) {
    return x + 100000 * y;
  }

  public void clear() {
    terrains.clear();
  }

  public Optional<TerrainType> tile(final int x, final int y) {
    final int key = id(x, y);
    if (terrains.containsKey(key)) {
      return Optional.of(terrains.get(key));
    } else {
      return Optional.empty();
    }
  }

  public void setTile(final int x, final int y, final TerrainType type) {
    final int key = id(x, y);
    terrains.put(key, type);
  }

  public int tilesCount() {
    return terrains.size();
  }
}
