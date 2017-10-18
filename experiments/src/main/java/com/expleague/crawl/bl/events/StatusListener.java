package com.expleague.crawl.bl.events;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface StatusListener extends SystemViewListener {
  void addStatus(final String messages);

  void removeStatus(final String messages);
}
