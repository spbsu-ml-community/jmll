package com.spbsu.crawl.bl.map.mapEvents;

public class ForgetMapEvent implements MapEvent {
  @Override
  public MapEventType event() {
    return MapEventType.FORGET_MAP;
  }
}
