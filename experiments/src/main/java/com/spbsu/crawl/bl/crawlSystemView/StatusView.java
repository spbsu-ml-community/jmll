package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.bl.events.StatusListener;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by noxoomo on 17/07/16.
 */
public class StatusView extends Subscribable.Stub<StatusListener> implements Subscribable<StatusListener> {
  private final Updater updater = new Updater();
  private Set<String> currentStatus = new HashSet<>();

  public Updater updater() {
    return updater;
  }

  class Updater {
    void updateStatus(final List<PlayerInfoMessage.PlayerStatus> statuses) {
      final Set<String> newStatus = statuses.stream().map(PlayerInfoMessage.PlayerStatus::text).collect(Collectors.toSet());

      currentStatus.forEach(oldStatus -> {
        if (!newStatus.contains(oldStatus)) {
          listeners().forEach(statusListener -> statusListener.removeStatus(oldStatus));
        }
      });

      newStatus.forEach(newStatusMessage -> {
        if (!currentStatus.contains(newStatusMessage)) {
          listeners().forEach(statusListener -> statusListener.addStatus(newStatusMessage));
        }
      });

      currentStatus = newStatus;
    }
  }
}
