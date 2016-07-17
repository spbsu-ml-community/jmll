package com.spbsu.crawl.bl.events;

import java.util.List;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface StatusListener extends SystemViewListener {
  void addStatus(final String messages);
  void removeStatus(final String messages);
}
