package com.expleague.crawl.data;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

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

  ObjectMapper mapper = new ObjectMapper();
  default String json() {
    final ObjectNode node = mapper.valueToTree(this);
    node.set("msg", new TextNode(type().name().toLowerCase()));
    try {
      return mapper.writeValueAsString(node);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }
}

