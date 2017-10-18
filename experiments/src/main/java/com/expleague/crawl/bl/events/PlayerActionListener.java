package com.expleague.crawl.bl.events;

import com.expleague.crawl.bl.Mob;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface PlayerActionListener  extends SystemViewListener {
  void action(final Mob.Action action);
  
}
