package com.spbsu.crawl.bl.map.props;

public class Obstruction implements CellProperty {
  private static Obstruction instance = new Obstruction();

  public static Obstruction instance() {
    return instance;
  }
}
