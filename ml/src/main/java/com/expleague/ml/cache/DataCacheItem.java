package com.expleague.ml.cache;

import de.schlichtherle.truezip.nio.file.TPath;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;

import java.io.*;
import java.nio.charset.StandardCharsets;

public interface DataCacheItem<T, D, Conf extends DataCacheConfig> {
  TPath getPath();

  void update(OutputStream os) throws IOException;
  T read(InputStream is) throws IOException;

  // used by descendants
  Logger getLogger();
  DataCache<D, Conf> getOwner();

  class Stub<T, D, Conf extends DataCacheConfig> implements DataCacheItem<T, D, Conf> {
    // These variables are set during component initialization by reflection
    private DataCache<D, Conf> owner;
    private Logger logger;

    private final String name;

    protected Stub(String name) {
      this.name = name;
    }

    public DataCache<D, Conf> getOwner() {
      return owner;
    }

    /**
     * @param in close must be handled by this function
     */
    public T read(InputStream in) throws IOException {
      return read(new InputStreamReader(in, StandardCharsets.UTF_8));
    }

    public T read(Reader reader) throws IOException {
      throw new UnsupportedOperationException();
    }

    public void update(OutputStream out) throws IOException {
      try (OutputStreamWriter writer = new OutputStreamWriter(out, StandardCharsets.UTF_8)) {
        update(writer);
      }
    }

    public void update(Writer out) throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public Logger getLogger() {
      return logger;
    }

    @SuppressWarnings("WeakerAccess")
    @NotNull
    public TPath getPath() {
      return new TPath(owner.getPath() + "/" + name);
    }
  }
}
