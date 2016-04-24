package com.spbsu.crawl.bl.map;

public class Unknown implements Cell {
  @Override
  public PassabilityType passability() {
    return PassabilityType.UNKNOWN;
  }

  @Override
  public CellType type() {
    return CellType.UNKNOWN;
  }
}
