package com.spbsu.crawl.bl.map.mapEvents;

import com.spbsu.crawl.bl.map.TerrainType;

public class CellMapEvent implements MapEvent {
  @Override
  public MapEventType type() {
    return eventType;
  }

  final int x;
  final int y;
  final MapEventType eventType;
  final TerrainType terrainType;


  public CellMapEvent(final int x, final int y,
                      final MapEventType eventType,
                      final TerrainType type) {
    this.x = x;
    this.y = y;
    this.eventType = eventType;
    this.terrainType = type;
  }

  public int x() {
    return x;
  }

  public int y() {
    return y;
  }

  public TerrainType terrainType() {
    return terrainType;
  }

  public MapEventType eventType() {
    return eventType;
  }
}

