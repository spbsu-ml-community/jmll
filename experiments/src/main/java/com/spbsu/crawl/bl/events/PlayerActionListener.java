package com.spbsu.crawl.bl.events;

import com.spbsu.crawl.bl.Mob;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface PlayerActionListener  extends SystemViewListener {
  void action(final Mob.Action action);
  
}
