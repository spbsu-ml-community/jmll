package com.spbsu.crawl;

import com.spbsu.crawl.bl.map.Level;
import com.spbsu.crawl.bl.map.Map;
import com.spbsu.crawl.bl.map.TerrainType;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.UpdateMapMessage;

/**
 * User: noxoomo
 */

public class MapBuilder {
  private Map map;

  UpdateMapMessage mapMessage;
  PlayerInfoMessage playerMessage;

  public MapBuilder() {
  }

  void setPlayerInfoMessage(PlayerInfoMessage message) {
    playerMessage = message;
    tryMapUpdate();
  }

  private void cellHandler(final UpdateMapCellMessage cellMessage, final Level level) {
    map.observeCell(level, cellMessage.getX(), cellMessage.getY(), TerrainType.fromMessage(cellMessage));
  }

  private void updateMapHandler(final UpdateMapMessage mapMessage, final Level level) {
    if (mapMessage.isForceFullRedraw()) {
      map.clear(level);
    }

    mapMessage.getCells().stream()
            .filter(cell -> cell.getDungeonFeatureType() != 0)
            .forEach(cellMessage -> cellHandler(cellMessage, level));
  }

  public void setMapMessage(final UpdateMapMessage message) {
    this.mapMessage = message;
    tryMapUpdate();
  }

  boolean tryMapUpdate() {
    if (mapMessage == null || playerMessage == null) {
      return false;
    }
    Level level = new Level(playerMessage.getDepth());
    updateMapHandler(mapMessage, level);
    mapMessage = null;
    playerMessage = null;
    return true;
  }

  Map map() {
    return map;
  }
}
