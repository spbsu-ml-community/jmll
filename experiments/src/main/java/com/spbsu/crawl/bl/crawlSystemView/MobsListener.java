package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.bl.Mob;

/**
 * User: Noxoomo
 * Date: 14.08.16
 * Time: 10:11
 */
public interface MobsListener {

  void observeMonster(final Mob mob);

  void lostMonster(final Mob mob);

}
