package com.spbsu.ml.io;

import java.io.IOException;
import java.io.StringWriter;


import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.spbsu.commons.func.types.ConversionPack;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.seq.CharSequenceReader;
import com.spbsu.ml.meta.items.QURLItem;

/**
 * User: solar
 * Date: 16.07.14
 * Time: 13:09
 */
public class QURLItem2CharSequenceConversionPack implements ConversionPack<QURLItem, CharSequence> {
  public static class To implements TypeConverter<QURLItem, CharSequence> {
    JsonFactory factory = new JsonFactory();
    final ObjectMapper mapper = new ObjectMapper(factory);
    @Override
    public CharSequence convert(final QURLItem from) {
      final StringWriter writer = new StringWriter();
      try {
        final JsonGenerator generator = factory.createGenerator(writer);
        generator.setCodec(mapper);
        generator.disable(JsonGenerator.Feature.QUOTE_FIELD_NAMES);
        generator.writeObject(from);
        return writer.toString();
      } catch (IOException e) {
        // never happen
        throw new RuntimeException(e);
      }
    }
  }
  public static class From implements TypeConverter<CharSequence, QURLItem> {
    JsonFactory factory = new JsonFactory();
    final ObjectMapper mapper = new ObjectMapper(factory);
    {
      factory.enable(JsonParser.Feature.ALLOW_UNQUOTED_FIELD_NAMES);
      factory.setCodec(mapper);
    }
    @Override
    public QURLItem convert(final CharSequence from) {
      try {
        final JsonParser parser = factory.createParser(new CharSequenceReader(from));
        return parser.readValueAs(QURLItem.class);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }
  @Override
  public Class<? extends TypeConverter<QURLItem, CharSequence>> to() {
    return To.class;
  }

  @Override
  public Class<? extends TypeConverter<CharSequence, QURLItem>> from() {
    return From.class;
  }
}
