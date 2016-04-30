package com.spbsu.crawl.bl.map;

import com.spbsu.crawl.bl.map.actions.CellAction;
import com.spbsu.crawl.bl.map.actions.GoDown;
import com.spbsu.crawl.bl.map.actions.GoUp;
import com.spbsu.crawl.bl.map.actions.StairsAction;
import com.spbsu.crawl.bl.map.props.CellProperty;
import com.spbsu.crawl.bl.map.props.Entrance;

import java.util.Set;
import java.util.stream.Stream;

public class Stair extends Floor {
  private Entrance entrance;
  private StairsAction action;

  public void update(final Stair cell) {
    entrance = cell.entrance;
    action = cell.action;
  }

  @Override
  public void merge(Cell cell) {
    super.merge(cell);
    entrance = ((Stair) cell).entrance;
    action = ((Stair) cell).action;
  }

  public Stair(int x, int y, Entrance entrance) {
    super(x, y);
    this.entrance = entrance;
    switch (entrance.type()) {
      case Up:
        this.action = new GoUp();
        break;
      case Down:
        this.action = new GoDown();
        break;
    }
  }

  @Override
  public Stream<CellAction> actions() {
    return Stream.concat(super.actions(), Stream.of(action));
  }

  @Override
  public Set<CellProperty> props() {
    Set<CellProperty> propsSet = super.props();
    propsSet.add(entrance);
    return propsSet;
  }


}
