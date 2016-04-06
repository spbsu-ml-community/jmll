package com.spbsu.crawl.data;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */

@JsonIgnoreProperties(ignoreUnknown = true)
public interface Message {

  default Protocol type() {
    return Stream.of(Protocol.values())
            .filter(t -> getClass().equals(t.clazz()))
            .findFirst()
            .orElse(null);
  }
}
