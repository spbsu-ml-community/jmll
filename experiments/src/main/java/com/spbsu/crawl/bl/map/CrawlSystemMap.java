package com.spbsu.crawl.bl.map;


import com.spbsu.crawl.bl.map.mapEvents.*;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.UpdateMapMessage;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class CrawlSystemMap {
  private Layer layer;
  private Level level = new Level("1");
  private List<MapEventListener> listeners;
  private Updater updater = new Updater();

  private void sendEvent(final MapEvent event) {
    for (MapEventListener listener : listeners) {
      listener.observe(event);
    }
  }

  public CrawlSystemMap() {
    layer = new Layer(level);
    listeners = new ArrayList<>();
  }

  public void subscribeToEvents(final MapEventListener listener) {
    listeners.add(listener);
  }

  public void observeCell(final int x, final int y,
                          final TerrainType type) {

    final Optional<TerrainType> terrain = layer.getCell(x, y);

    if (terrain.isPresent()) {
      if (terrain.get() != type) {
        layer.setCell(x, y, type);
        sendEvent(new CellMapEvent(x, y, MapEventType.CHANGED_CELL, type));
      }
    } else {
      layer.setCell(x, y, type);
      sendEvent(new CellMapEvent(x, y, MapEventType.OBSERVED_CELL, type));
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
    Level current = new Level("1");

    public Updater() {
    }

    public void message(PlayerInfoMessage message) {
      playerMessage = message;
      tryMapUpdate();
    }

    private void clear() {
      if (playerMessage.depth() != null) {
        current = new Level(playerMessage.depth());
      }
      layer = new Layer(level);
      sendEvent(new ChangeSystemMapEvent(level));
    }

    private void cellHandler(final UpdateMapCellMessage cellMessage) {
      observeCell(cellMessage.x(), cellMessage.y(), TerrainType.fromMessage(cellMessage));
    }

    public void message(final UpdateMapMessage message) {
      this.mapMessage = message;
      tryMapUpdate();
    }

    void tryMapUpdate() {
      if (mapMessage == null || playerMessage == null) {
        return;
      }

      if (mapMessage.isForceFullRedraw()) {
        clear();
      }

      List<UpdateMapCellMessage> cells = mapMessage.getCells();

      for (int i = 1; i < cells.size(); ++i) {
        if (cells.get(i).x() == Integer.MAX_VALUE) {
          cells.get(i).setPoint(cells.get(i - 1).x() + 1, cells.get(i - 1).y());
        }
      }

      cells.stream()
              .filter(cell -> cell.getDungeonFeatureType() != 0)
              .forEach(this::cellHandler);

      mapMessage = null;
      playerMessage = null;
    }
  }
}



