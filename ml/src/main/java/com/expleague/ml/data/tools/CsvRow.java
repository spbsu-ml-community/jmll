package com.expleague.ml.data.tools;

import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import org.apache.commons.lang3.time.DateParser;
import org.apache.commons.lang3.time.FastDateFormat;

import java.text.ParseException;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

@SuppressWarnings("OptionalGetWithoutIsPresent")
public interface CsvRow extends Function<String, Optional<CharSeq>>, Cloneable {
  CsvRow names();
  CsvRow clone();

  CharSeq at(int i);

  default double asDouble(String name) {
    final Optional<CharSeq> apply = apply(name);
    if (apply.isPresent())
      return CharSeqTools.parseDouble(apply.get());
    else
      throw new IllegalArgumentException("Unable to find non-empty double field " + name);
  }

  default double asDouble(String name, double defaultV) {
    try {
      return apply(name).map(CharSeqTools::parseDouble).orElse(defaultV);
    }
    catch (NumberFormatException ignore) {
      return defaultV;
    }
  }

  default int asInt(String name) {
    final Optional<CharSeq> apply = apply(name);
    if (apply.isPresent())
      return CharSeqTools.parseInt(apply.get());
    else
      throw new IllegalArgumentException("Unable to find non-empty integer field " + name);
  }

  default int asInt(String name, int defaultV) {
    try {
      return apply(name).map(CharSeqTools::parseInt).orElse(defaultV);
    }
    catch (NumberFormatException ignore) {
      return defaultV;
    }
  }

  default boolean asBool(String name) {
    final Optional<CharSeq> apply = apply(name);
    if (apply.isPresent()) {
      CharSeq val = apply.get();
      //noinspection EqualsBetweenInconvertibleTypes
      return val.equals("true") || val.equals("1") || val.equals("da") || val.equals("yes");
    }
    else
      throw new IllegalArgumentException("Unable to find non-empty integer field " + name);
  }

  default boolean asBool(String name, boolean defaultV) {
    //noinspection EqualsBetweenInconvertibleTypes
    return apply(name).map(val -> val.equals("true") || val.equals("1") || val.equals("da") || val.equals("yes")).orElse(defaultV);
  }

  default long asLong(String name) {
    final Optional<CharSeq> apply = apply(name);
    if (apply.isPresent())
      return CharSeqTools.parseLong(apply.get());
    else
      throw new IllegalArgumentException("Unable to find non-empty long field " + name);
  }

  default long asLong(String name, long defaultV) {
    try {
      return apply(name).map(CharSeqTools::parseLong).orElse(defaultV);
    }
    catch (NumberFormatException ignore) {
      return defaultV;
    }
  }

  Map<String, DateParser> dateParsers = new HashMap<>();
  default Date asDate(String name, String pattern) {
    final DateParser parser = dateParsers.compute(pattern, (p, v) -> v != null ? v : FastDateFormat.getInstance(p));
    try {
      Optional<CharSeq> apply = apply(name);
      if (apply.isPresent())
        return parser.parse(apply.get().toString());
      else
        throw new IllegalArgumentException("Unable to find non-empty date field " + name);
    }
    catch (ParseException e) {
      throw new RuntimeException(e);
    }
  }

  default String asString(String name) {
    return apply(name).map(CharSeq::toString).orElse(null);
  }
  default String asString(String name, String defaultValue) {
    return apply(name).map(CharSeq::toString).orElse(defaultValue);
  }

  default CharSeq at(String type) {
    return apply(type).orElse(null);
  }
}
