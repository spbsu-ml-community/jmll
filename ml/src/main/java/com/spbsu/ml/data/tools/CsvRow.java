package com.spbsu.ml.data.tools;

import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import org.apache.commons.lang3.time.DateParser;
import org.apache.commons.lang3.time.FastDateFormat;

import java.text.ParseException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

@SuppressWarnings("OptionalGetWithoutIsPresent")
public interface CsvRow extends Function<String, Optional<CharSeq>> {
  CharSeq at(int i);
  CsvRow names();

  default double asDouble(String header) {
    return CharSeqTools.parseDouble(apply(header).get());
  }

  default double asDouble(String name, double defaultV) {
    try {
      return apply(name).map(CharSeqTools::parseDouble).orElse(defaultV);
    }
    catch (IllegalArgumentException iae) {
      return defaultV;
    }
  }

  default int asInt(String header) {
    return CharSeqTools.parseInt(apply(header).get());
  }

  default int asInt(String name, int defaultV) {
    try {
      return apply(name).map(CharSeqTools::parseInt).orElse(defaultV);
    }
    catch (IllegalArgumentException iae) {
      return defaultV;
    }
  }

  Map<String, DateParser> dateParsers = new HashMap<>();
  default Date asDate(String name, String pattern) {
    final DateParser parser = dateParsers.compute(pattern, (p, v) -> v != null ? v : FastDateFormat.getInstance(p));
    try {
      return parser.parse(apply(name).get().toString());
    }
    catch (ParseException e) {
      throw new RuntimeException(e);
    }
  }

  default String asString(String name) {
    return apply(name).get().toString();
  }

  default long asLong(String name) {
    return CharSeqTools.parseLong(apply(name).get());
  };

  default long asLong(String name, long defaultV) {
    return apply(name).map(CharSeqTools::parseLong).orElse(defaultV);
  };

  default CharSeq at(String type) {
    return apply(type).get();
  }
}
