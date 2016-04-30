package com.spbsu.crawl.bl.map;

import com.spbsu.crawl.bl.map.props.CellProperty;
import com.spbsu.crawl.bl.map.props.Obstruction;

import java.util.Collections;
import java.util.Set;

public class Wall extends Cell {
  private final static Set<CellProperty> wallProps = Collections.singleton(Obstruction.instance());

  public Wall(int x, int y) {
    super(x, y);
  }

  @Override
  public void merge(Cell cell) {

  }

  @Override
  final public Set<CellProperty> props() {
    return wallProps;
  }
}
