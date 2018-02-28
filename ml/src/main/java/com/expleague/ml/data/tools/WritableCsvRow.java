package com.expleague.ml.data.tools;

import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqBuilder;
import com.expleague.commons.seq.CharSeqTools;
import gnu.trove.map.TObjectIntMap;

import java.util.Arrays;
import java.util.Optional;

public class WritableCsvRow implements CsvRow {
  private final CharSeq[] split;
  private final TObjectIntMap<String> names;

  public WritableCsvRow(CharSeq[] split, TObjectIntMap<String> names) {
    this.split = split;
    this.names = names;
  }

  @Override
  public CharSeq at(int i) {
    return split[i];
  }

  public void set(int i, CharSeq seq) {
    split[i] = seq;
  }

  public void set(String name, CharSeq seq) {
    final int index = names.get(name);
    if (index == 0)
      throw new RuntimeException("Stream does not contain required column '" + name + "'!");
    split[index - 1] = seq;
  }

  public void set(String name, int v) {
    set(name, CharSeq.create(Integer.toString(v)));
  }

  public void set(String name, long v) {
    set(name, CharSeq.create(Long.toString(v)));
  }

  public void set(String name, double v) {
    set(name, CharSeq.create(Double.toString(v)));
  }

  public void set(String name, float v) {
    set(name, CharSeq.create(Float.toString(v)));
  }

  public void set(String name, boolean v) {
    set(name, CharSeq.create(Boolean.toString(v)));
  }

  public void set(String name, String v) {
    set(name, CharSeq.create(v));
  }

  @Override
  public CsvRow names() {
    final CharSeq[] names = new CharSeq[Arrays.stream(this.names.values()).max().orElse(0)];
    Arrays.fill(names, CharSeq.create("duplicate"));
    this.names.forEachEntry((name, index) -> {
      names[index - 1] = CharSeq.create(name);
      return true;
    });
    return new WritableCsvRow(names, this.names);
  }

  @Override
  public Optional<CharSeq> apply(String name) {
    final int index = names.get(name);
    if (index == 0)
      throw new RuntimeException("Stream does not contain required column '" + name + "'!");
    final CharSeq part = split[index - 1];
    return part.length() > 0 ? Optional.of(part) : Optional.empty();
  }

  @Override
  public String toString() {
    final CharSeqBuilder builder = new CharSeqBuilder();
    for (int i = 0; i < split.length; i++) {
      builder.append('"').append(CharSeqTools.replace(split[i], "\"", "\"\"")).append('"');
      if (i < split.length - 1)
        builder.append(',');
    }
    return builder.toString();
  }

  public CsvRow clone() {
    CharSeq[] split = new CharSeq[this.split.length];
    for (int i = 0; i < split.length; i++) {
      split[i] = CharSeq.intern(this.split[i]);
    }
    return new WritableCsvRow(split, names);
  }
}
