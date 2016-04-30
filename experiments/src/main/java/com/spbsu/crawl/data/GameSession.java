package com.spbsu.crawl.data;

import com.spbsu.crawl.bl.map.Cell;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession {
  Hero.Race race();

  Hero.Spec spec();

  void heroPosition(int x, int y);

  void setDungeonLevel(int level);

  void resetCells();

  void observeCell(Cell cell);
}
