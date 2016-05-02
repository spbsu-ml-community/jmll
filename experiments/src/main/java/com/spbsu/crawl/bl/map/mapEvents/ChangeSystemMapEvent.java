package com.spbsu.crawl.bl.map.mapEvents;

import com.spbsu.crawl.bl.map.Level;

public class ChangeSystemMapEvent implements MapEvent {
  private Level level;

  public ChangeSystemMapEvent(Level level) {
    this.level = level;
  }

  public Level level() {
    return level;
  }

  @Override
  public MapEventType type() {
    return MapEventType.SYSTEM_MAP_CHANGED;
  }
}
