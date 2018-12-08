package com.expleague.ml.cache;

public class MultipartDataCacheItem<T, D, Conf extends DataCacheConfig> extends DataCacheItem.Stub<T, D, Conf> {
  public MultipartDataCacheItem(String name, Class<? extends DataCacheItem> partClass) {
    super(name);
    try {
      //noinspection JavaReflectionMemberAccess
      partClass.getConstructor(String.class, int.class);
    } catch (NoSuchMethodException e) {
      throw new RuntimeException("Unable to create multipart cache item: no component constructor with path and index (String, int) parameters", e);
    }
  }
}

