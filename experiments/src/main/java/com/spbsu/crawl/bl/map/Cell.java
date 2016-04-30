package com.spbsu.crawl.bl.map;

import com.spbsu.crawl.bl.map.actions.CellAction;
import com.spbsu.crawl.bl.map.props.CellProperty;

import java.util.Collections;
import java.util.Set;
import java.util.stream.Stream;


public abstract class Cell {
  private int x;
  private int y;

  public Cell(final int x, final int y) {
    this.x = x;
    this.y = y;
  }

  final int x() {
    return x;
  }

  final int y() {
    return y;
  }

  public Stream<CellAction> actions() {
    return Stream.empty();
  }

  public abstract void merge(final Cell cell);

  public Set<CellProperty> props() {
    return Collections.emptySet();
  }
}



