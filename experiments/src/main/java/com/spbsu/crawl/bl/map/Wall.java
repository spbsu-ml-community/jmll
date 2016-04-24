package com.spbsu.crawl.bl.map;

public class Wall implements Cell {

  @Override
  final public PassabilityType passability() {
    return PassabilityType.OBSTRUCTION;
  }
}
