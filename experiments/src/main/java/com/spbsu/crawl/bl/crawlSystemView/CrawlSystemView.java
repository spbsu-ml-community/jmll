package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.UpdateMapMessage;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

import java.util.List;

/**
 * Created by noxoomo on 14/07/16.
 */

public class CrawlSystemView {
  private final CrawlHeroView heroView = new CrawlHeroView();
  private final CrawlSystemMapView mapView = new CrawlSystemMapView();
  private final CrawlTimeView timeView = new CrawlTimeView();
  private Updater updater = new Updater();

  public CrawlHeroView heroView() {
    return heroView;
  }

  public CrawlSystemMapView mapView() {
    return mapView;
  }

  public CrawlTimeView timeView() {
    return timeView;
  }

  public Updater updater() {
    return updater;

  }

  public class Updater {
    private PlayerInfoMessage lastPlayerMessage;

    public void message(final PlayerInfoMessage playerMessage) {
      this.lastPlayerMessage = playerMessage;
      updatePlayer();
      if (!EmptyFieldsDefault.isEmpty(playerMessage.turn())) {
        timeView.updater().setTime(playerMessage.time(), playerMessage.turn());
      }
    }

    public void message(final UpdateMapMessage message) {
      final List<UpdateMapCellMessage> cells = message.getCells();
      for (int i = 1; i < cells.size(); ++i) {
        if (EmptyFieldsDefault.isEmpty(cells.get(i).x())) {
          cells.get(i).setPoint(cells.get(i - 1).x() + 1, cells.get(i - 1).y());
        }
      }
      updateMap(message);
    }

    private void updateMap(final UpdateMapMessage mapMessage) {
      if (mapMessage.isForceFullRedraw()) {
        if (EmptyFieldsDefault.notEmpty(lastPlayerMessage.depth())) {
          mapView.updater().updateLevel(lastPlayerMessage.depth());
        }
        mapView.updater().clear();
      }
      mapView.updater().update(mapMessage.getCells());

    }

    private void updatePlayer() {
      heroView.updater().message(lastPlayerMessage);
    }

  }
}
