package com.expleague.crawl.bl.crawlSystemView;

import com.expleague.crawl.bl.Mob;
import com.expleague.crawl.bl.events.PlayerActionListener;

/**
 * Created by noxoomo on 16/07/16.
 */
public class PlayerActionView extends Subscribable.Stub<PlayerActionListener> implements Subscribable<PlayerActionListener> {
  public void action(final Mob.Action action) {
    listeners().forEach(playerActionListener -> playerActionListener.action(action));
  }
}
