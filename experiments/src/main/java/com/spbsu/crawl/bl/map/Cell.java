package com.spbsu.crawl.bl.map;

import java.util.stream.Stream;

public interface Cell {

  PassabilityType passability();

  default CellType type() {
    return Stream.of(CellType.values())
            .filter(t -> getClass().equals(t.clazz()))
            .findFirst()
            .orElse(null);
  }
}

