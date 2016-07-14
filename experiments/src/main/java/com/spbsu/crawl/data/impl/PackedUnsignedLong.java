package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import java.io.IOException;

/**
 * Created by noxoomo on 14/07/16.
 */
@JsonDeserialize(using = PackedUnsignedLong.PackedUnsignedLongDeserializer.class)
public class PackedUnsignedLong {
  private final int lowerbits;
  private final int upperbits;

  public PackedUnsignedLong(final int lowerbits,
                            final int upperbits) {
    this.lowerbits = lowerbits;
    this.upperbits = upperbits;
  }

  static class PackedUnsignedLongDeserializer extends JsonDeserializer<PackedUnsignedLong> {

    @Override
    public PackedUnsignedLong deserialize(JsonParser jsonParser,
                                          DeserializationContext deserializationContext) throws IOException, JsonProcessingException {

      JsonNode node = jsonParser.getCodec().readTree(jsonParser);
      final int lowerbits;
      final int upperbits;

      if (node.isArray()) {
        lowerbits = node.get(0).asInt();
        upperbits = node.get(1).asInt();
      } else {
        lowerbits = node.asInt();
        upperbits = 0;
      }
      return new PackedUnsignedLong(lowerbits, upperbits);
    }
  }
}
