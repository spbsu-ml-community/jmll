package com.spbsu.crawl.bl.crawlSystemView;


import com.spbsu.crawl.bl.map.Layer;
import com.spbsu.crawl.bl.map.TerrainType;
import com.spbsu.crawl.bl.events.MapListener;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class CrawlSystemMapView {
  private Layer layer = new Layer();
  private List<MapListener> listeners = new ArrayList<>();
  private Updater updater = new Updater();

  public void subscribe(final MapListener listener) {
    listeners.add(listener);
  }

  public void observeCell(final int x, final int y,
                          final TerrainType type) {
    final Optional<TerrainType> terrain = layer.tile(x, y);

    if (terrain.isPresent()) {
      if (terrain.get() != type) {
        layer.setTile(x, y, type);
        listeners.stream().forEach(lst -> lst.tile(x, y, type));
      }
    } else {
      layer.setTile(x, y, type);
      listeners.stream().forEach(lst -> lst.tile(x, y, type));
    }
  }


  public Updater updater() {
    return updater;
  }

  public int knownCells() {
    return layer.tilesCount();
  }

  /**
   * User: noxoomo
   */
  public class Updater {
    public Updater() {
    }

    void updateLevel(final String newLevel) {
      if (EmptyFieldsDefault.notEmpty(newLevel)) {
        listeners.stream().forEach(lst -> lst.changeLevel(newLevel));
      }
    }

    void clear() {
      layer.clear();
      listeners.stream().forEach(MapListener::resetPosition);
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



