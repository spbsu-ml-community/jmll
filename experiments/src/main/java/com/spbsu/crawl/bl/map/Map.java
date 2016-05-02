package com.spbsu.crawl.bl.map;


import com.spbsu.crawl.bl.map.mapEvents.*;
import gnu.trove.map.hash.THashMap;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class Map {
  private THashMap<Level, Layer> layers;
  private List<MapEventListener> listeners;

  private void sendEvent(final MapEvent event) {
    for (MapEventListener listener : listeners) {
      listener.observe(event);
    }
  }

  public Map() {
    layers = new THashMap<>();
    listeners = new ArrayList<>();
  }

  public void subscribeToEvents(final MapEventListener listener) {
    listeners.add(listener);
  }

  public void observeCell(final Level level, final int x, final int y,
                          final TerrainType type) {
    if (!layers.containsKey(level)) {
      layers.put(level, new Layer(level));
    }

    final Layer layer = layers.get(level);
    final Optional<TerrainType> terrain = layer.getCell(x, y);

    if (terrain.isPresent()) {
      if (terrain.get() != type) {
        layer.setCell(x, y, type);
        sendEvent(new CellMapEvent(x, y, MapEventType.CHANGED_CELL));
      }
    } else {
      layer.setCell(x, y, type);
      sendEvent(new CellMapEvent(x, y, MapEventType.OBSERVED_CELL));
    }
  }

  public void clear(final Level level) {
    if (layers.containsKey(level)) {
      layers.get(level).clear();
      sendEvent(new ForgetMapEvent());
    }
  }

}



