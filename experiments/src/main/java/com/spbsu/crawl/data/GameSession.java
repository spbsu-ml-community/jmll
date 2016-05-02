package com.spbsu.crawl.data;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession  {
  Hero.Race race();

  Hero.Spec spec();

  void heroPosition(int x, int y);

  void setDungeonLevel(int level);

}
