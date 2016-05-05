package com.spbsu.crawl.bl.map;


import com.spbsu.crawl.data.MapListener;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.UpdateMapMessage;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class CrawlSystemMap {
  private Layer layer = new Layer();
  private List<MapListener> listeners = new ArrayList<>();
  private Updater updater = new Updater();

  public void subscribeToEvents(final MapListener listener) {
    listeners.add(listener);
  }

  public void observeCell(final int x, final int y, final TerrainType type) {
    final Optional<TerrainType> terrain = layer.tile(x, y);

    if (terrain.isPresent()) {
      if (terrain.get() != type) {
        layer.setTile(x, y, type);
        listeners.stream().forEach(lst -> lst.tile(x, y, type));
      }
    }
    else {
      layer.setTile(x, y, type);
      listeners.stream().forEach(lst -> lst.tile(x, y, type));
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

    private void clear() {
      if (playerMessage.depth() != null) {
        listeners.stream().forEach(lst -> lst.changeLevel(playerMessage.depth()));
      }
      layer.clear();
      listeners.stream().forEach(MapListener::resetPosition);
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



