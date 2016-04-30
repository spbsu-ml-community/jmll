package com.spbsu.crawl.bl.map;

import com.spbsu.crawl.bl.map.actions.CellAction;
import com.spbsu.crawl.bl.map.actions.GoDown;
import com.spbsu.crawl.bl.map.actions.GoUp;
import com.spbsu.crawl.bl.map.actions.StairsAction;

import java.util.stream.Stream;

public class Stair extends Floor {
  public enum Direction {
    UP,
    DOWN
  }

  private Direction direction;
  private StairsAction action;

  public void update(final Stair cell) {
    direction = cell.direction;
    action = cell.action;
  }

  @Override
  public void merge(Cell cell) {
    super.merge(cell);
    direction = ((Stair) cell).direction;
    action = ((Stair) cell).action;
  }

  public Stair(int x, int y, Direction direction) {
    super(x, y);
    this.direction = direction;
    switch (direction) {
      case UP:
        this.action = new GoUp();
        break;
      case DOWN:
        this.action = new GoDown();
        break;
    }
  }

  @Override
  public Stream<CellAction> actions() {
    return Stream.concat(super.actions(), Stream.of(action));
  }

}
