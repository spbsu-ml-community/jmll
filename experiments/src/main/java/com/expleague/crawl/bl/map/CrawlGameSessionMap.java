package com.expleague.crawl.bl.map;


import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class CrawlGameSessionMap {
  private final Map<String, Layer<TerrainType>> layers = new HashMap<>();
  private String currentLevel;


  public Optional<TerrainType> terrainOnCurrentLevel(int x, int y) {
    return layers.get(currentLevel).item(x, y);
  }

  public void changeLevel(String id) {
    currentLevel = id;
  }

  public void resetPosition() {
    currentLayer().clear();
  }

  public void tile(final int x, final int y, final TerrainType type) {
    currentLayer().setItem(x, y, type);
  }

  private Layer<TerrainType> currentLayer() {
    return layers.compute(currentLevel, (level, layer) -> {
      if (layer == null) {
        layers.put(level, layer = new Layer<>());
      }
      return layer;
    });
  }
}



