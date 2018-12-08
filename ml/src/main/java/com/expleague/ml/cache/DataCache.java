package com.expleague.ml.cache;

import java.nio.file.Path;
import java.util.Arrays;

public interface DataCache<D, Conf extends DataCacheConfig> {
  Class<? extends DataCacheItem>[] available();
  default boolean isAvailable(Class<? extends DataCacheItem> part) {
    return Arrays.stream(available()).anyMatch(part::equals);
  }

  boolean contains(Class<? extends DataCacheItem> part);
  void update(Class<? extends DataCacheItem> part);
  <T, P extends DataCacheItem<T, ? super D, ?>> T get(Class<P> part);

  Path getPath();

  // Shared components
  D getDataManager();
  Conf getConfig();
}
