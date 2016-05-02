package com.spbsu.crawl.bl.map;


import com.spbsu.crawl.bl.map.mapEvents.CellMapEvent;
import com.spbsu.crawl.bl.map.mapEvents.ChangeSystemMapEvent;
import com.spbsu.crawl.bl.map.mapEvents.MapEvent;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class CrawlGameSessionMap {
  private final Map<Level, Layer> layers = new HashMap<>();
  private Level currentLevel = new Level("1");


  private void event(final ChangeSystemMapEvent event) {
    currentLevel = event.level();
    if (layers.containsKey(currentLevel)) {
      layers.get(currentLevel).clear();
    } else {
      layers.put(currentLevel, new Layer(currentLevel));
    }
  }

  private void event(final CellMapEvent event) {
    layers.get(currentLevel).setCell(event.x(), event.y(), event.terrainType());
  }

  public void systemMapEvent(final MapEvent mapEvent) {
    if (mapEvent instanceof CellMapEvent) {
      event((CellMapEvent) mapEvent);
    } else if (mapEvent instanceof ChangeSystemMapEvent) {
      event((ChangeSystemMapEvent) mapEvent);
    }
  }

  public Optional<TerrainType> terrainOnCurrentLevel(int x, int y) {
    return layers.get(currentLevel).getCell(x, y);
  }
}



