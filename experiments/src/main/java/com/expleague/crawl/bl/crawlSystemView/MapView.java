package com.expleague.crawl.bl.crawlSystemView;


import com.expleague.crawl.bl.events.MapListener;
import com.expleague.crawl.bl.map.Position;
import com.expleague.crawl.bl.map.PositionManager;
import com.expleague.crawl.bl.map.TerrainType;
import com.expleague.crawl.data.impl.UpdateMapCellMessage;
import com.expleague.crawl.data.impl.system.EmptyFieldsDefault;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MapView extends Subscribable.Stub<MapListener> implements Subscribable<MapListener> {
  private final PositionManager positionManager;
  private Map<Position, TerrainType> layer = new HashMap<>();
  private Updater updater = new Updater();

  public MapView(final PositionManager positionManager) {
    this.positionManager = positionManager;
  }


  public void observeCell(final int x, final int y,
                          final TerrainType type) {
    final Position position = positionManager.getOrCreate(x, y);
    TerrainType terrain = layer.getOrDefault(position, null);
    if (terrain != type) {
      layer.put(position, type);
      listeners().forEach(lst -> lst.tile(position, type));
    }
  }


  public Updater updater() {
    return updater;
  }

  public int knownCells() {
    return layer.size();
  }

  /**
   * User: noxoomo
   */
  public class Updater {
    public Updater() {
    }

    void updateLevel(final String newLevel) {
      if (EmptyFieldsDefault.notEmpty(newLevel)) {
        listeners().forEach(lst -> lst.changeLevel(newLevel));
      }
    }

    void clear() {
      layer.clear();
      listeners().forEach(MapListener::resetPosition);
    }

    private void cellHandler(final UpdateMapCellMessage cellMessage) {
      observeCell(cellMessage.x(), cellMessage.y(), TerrainType.fromMessage(cellMessage));
    }

    void update(final List<UpdateMapCellMessage> cells) {
      cells.stream()
              .filter(cell -> cell.getDungeonFeatureType() != 0)
              .forEach(this::cellHandler);
    }
  }
}



