package com.spbsu.crawl.bl.map;


import com.spbsu.crawl.bl.map.mapEvents.*;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.UpdateMapMessage;

import java.util.*;

public class CrawlSystemMap {
  private Map<Level, Layer> layers;
  private List<MapEventListener> listeners;
  private Updater updater = new Updater();

  private void sendEvent(final MapEvent event) {
    for (MapEventListener listener : listeners) {
      listener.observe(event);
    }
  }

  public CrawlSystemMap() {
    layers = new HashMap<>();
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

  public Updater updater() {
    return updater;
  }

  /**
   * User: noxoomo
   */
  public class Updater {
    UpdateMapMessage mapMessage;
    PlayerInfoMessage playerMessage;

    public Updater() {
    }

    public void message(PlayerInfoMessage message) {
      playerMessage = message;
      tryMapUpdate();
    }

    private void cellHandler(final UpdateMapCellMessage cellMessage, final Level level) {
      observeCell(level, cellMessage.getX(), cellMessage.getY(), TerrainType.fromMessage(cellMessage));
    }

    private void updateMapHandler(final UpdateMapMessage mapMessage, final Level level) {
      if (mapMessage.isForceFullRedraw()) {
        clear(level);
      }

      mapMessage.getCells().stream()
              .filter(cell -> cell.getDungeonFeatureType() != 0)
              .forEach(cellMessage -> cellHandler(cellMessage, level));
    }

    public void message(final UpdateMapMessage message) {
      this.mapMessage = message;
      tryMapUpdate();
    }

    Level current = new Level("1");
    boolean tryMapUpdate() {
      if (mapMessage == null || playerMessage == null) {
        return false;
      }
      if (playerMessage.getDepth() != null)
        current = new Level(playerMessage.getDepth());
      updateMapHandler(mapMessage, current);
      mapMessage = null;
      playerMessage = null;
      return true;
    }
  }
}



