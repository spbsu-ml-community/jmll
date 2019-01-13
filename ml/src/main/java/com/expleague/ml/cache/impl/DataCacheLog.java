package com.expleague.ml.cache.impl;

import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.cache.DataCacheConfig;
import com.expleague.ml.cache.DataCacheItem;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Reader;
import java.util.stream.Stream;

public class DataCacheLog extends DataCacheItem.Stub<Stream<CharSeq>, Object, DataCacheConfig> {
  public DataCacheLog() {
    super("log.txt");
  }

  @Override
  public void update(OutputStream out) throws IOException {
    throw new UnsupportedOperationException("Not intended to be called");
  }

  @Override
  public Stream<CharSeq> read(Reader reader) throws IOException {
    return CharSeqTools.lines(reader, false);
  }
}
