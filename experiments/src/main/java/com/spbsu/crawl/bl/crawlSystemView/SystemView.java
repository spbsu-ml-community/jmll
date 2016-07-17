package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.bl.events.*;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import com.spbsu.crawl.data.impl.UpdateMapMessage;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

import java.util.List;

/**
 * Created by noxoomo on 14/07/16.
 */

public class SystemView implements Subscribable<SystemViewListener> {
  private final HeroView heroView = new HeroView();
  private final MapView mapView = new MapView();
  private final PlayerActionView playerActionView = new PlayerActionView();
  private final InventoryView inventoryView = new InventoryView();
  private final StatusView statusView = new StatusView();
  private Updater updater = new Updater();

  public HeroView heroView() {
    return heroView;
  }

  public MapView mapView() {
    return mapView;
  }

  public PlayerActionView playerActionView() {
    return playerActionView;
  }

  public StatusView statusView() {
    return statusView;
  }

  public InventoryView inventoryView() {
    return inventoryView;
  }

  public Updater updater() {
    return updater;
  }


  public class Updater {
    private PlayerInfoMessage lastPlayerMessage;

    public void message(final PlayerInfoMessage playerMessage) {
      //for map info
      this.lastPlayerMessage = playerMessage;
      //status
      if (EmptyFieldsDefault.notEmpty(playerMessage.statuses())) {
        statusView().updater().updateStatus(playerMessage.statuses());
      }
      //items
      if (EmptyFieldsDefault.notEmpty(playerMessage.items())) {
        playerMessage.items().forEach((id, value) -> inventoryView().updater().item(Integer.valueOf(id), value));
      }
      //player
      heroView.updater().message(playerMessage);

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

  }

  public void subscribe(final SystemViewListener listener) {
    if (listener instanceof MapListener) {
      mapView().subscribe((MapListener) listener);
    }
    if (listener instanceof HeroListener) {
      heroView().subscribe((HeroListener) listener);
    }
    if (listener instanceof PlayerActionListener) {
      playerActionView().subscribe((PlayerActionListener) listener);
    }
    if (listener instanceof InventoryListener) {
      inventoryView().subscribe((InventoryListener) listener);
    }
    if (listener instanceof StatusListener) {
      statusView().subscribe((StatusListener) listener);
    }
  }
}
