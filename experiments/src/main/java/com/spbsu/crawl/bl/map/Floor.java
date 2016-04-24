package com.spbsu.crawl.bl.map;

public class Floor implements Cell {
  public PassabilityType passability() {
    return PassabilityType.PASSABLE;
  }
}
