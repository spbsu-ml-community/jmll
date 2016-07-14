package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.bl.events.TurnListener;

import java.util.ArrayList;
import java.util.List;

public class CrawlTimeView {
  private int time = 0;
  private int turn = 0;
  private List<TurnListener> listeners = new ArrayList<>();
  private Updater updater = new Updater();


  public void subscribe(final TurnListener listener) {
    listeners.add(listener);
  }

  public Updater updater() {
    return updater;
  }

  class Updater {
    void setTime(int newTime, int newTurn) {
      boolean update = newTurn != turn;
      time = newTime;
      turn = newTurn;
      if (update) {
        listeners.forEach(TurnListener::nextTurn);
      }
    }
  }
}
