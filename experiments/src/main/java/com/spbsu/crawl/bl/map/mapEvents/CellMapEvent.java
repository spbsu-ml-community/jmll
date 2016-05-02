package com.spbsu.crawl.bl.map.mapEvents;

public class CellMapEvent implements MapEvent {
  @Override
  public MapEventType event() {
    return eventType;
  }

  final int x;
  final int y;
  final MapEventType eventType;

  public CellMapEvent(int x, int y, MapEventType eventType) {
    this.x = x;
    this.y = y;
    this.eventType = eventType;
  }
}

