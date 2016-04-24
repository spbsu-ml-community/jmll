package com.spbsu.crawl.bl.map;

public class Other implements Cell {
  private PassabilityType passability = PassabilityType.UNKNOWN;

  public void passability(PassabilityType passability) {
    this.passability = passability;
  }

  @Override
  public PassabilityType passability() {
    return passability;
  }
}
