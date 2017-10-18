package com.expleague.crawl.bl.events;

/**
 * Created by noxoomo on 14/07/16.
 */
public interface HeroListener extends SystemViewListener {

  void heroPosition(int x, int y);

  void hp(final int hp);
}
