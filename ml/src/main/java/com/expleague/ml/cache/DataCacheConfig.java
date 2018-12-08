package com.expleague.ml.cache;

import com.expleague.ml.cache.impl.DataCacheImpl;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.stream.Stream;

@SuppressWarnings({"WeakerAccess", "unused"})
public class DataCacheConfig {
  @JsonProperty
  protected String id;

  @JsonProperty
  protected String type; // needed for pool type detection

  @JsonProperty
  private Map<String, String[]> dependencies = new HashMap<>();

  @JsonProperty
  private Map<String, Date> updated = new HashMap<>();

  @JsonIgnore
  protected DataCache owner;

  public String getId() {
    return id;
  }

  public String getType() {
    return type;
  }

  public void notifyUpdated(String name) {
    DataCacheImpl owner = (DataCacheImpl) this.owner;
    if (owner == null) {
      return;
    }
    updated.put(name, new Date());
    owner.flushConfig();
  }

  public void notifyRead(String name) {
    DataCacheImpl owner = (DataCacheImpl) this.owner;
    owner.propertyRead(name);
  }

  public void updateDependencies(Class<? extends DataCacheItem> component, String[] dependsOn) {
    Arrays.sort(dependsOn);
    if (Arrays.equals(dependencies.get(component.getName()), dependsOn)) {
      return;
    }

    dependencies.put(component.getName(), dependsOn);
  }

  public String[] dependencies(Class component) {
    Set<String> deepDeps = new HashSet<>();
    List<String> tasks = new ArrayList<>();
    Set<String> done = new HashSet<>();
    tasks.add(component.getName());
    while (!tasks.isEmpty()) {
      String next = tasks.remove(tasks.size() - 1);
      done.add(next);
      Stream.of(dependencies.getOrDefault(next, new String[0]))
          .peek(deepDeps::add)
          .filter(c -> !done.contains(c))
          .forEach(tasks::add);
    }
    return deepDeps.toArray(new String[deepDeps.size()]);
  }

  @Nullable
  public Date updateTime(String property) {
    try {
      if (!owner.contains(Class.forName(property)))
        return new Date();
    }
    catch (ClassNotFoundException ignore) {
    }
    return updated.get(property);
  }

  @Nullable
  public Date updateTime(Class<? extends DataCacheItem> property) {
    return updated.get(property.getName());
  }

  public boolean isUpToDate(Class<? extends DataCacheItem> item) {
    return this.isUpToDate(item, new ArrayList<>());
  }

  public boolean isUpToDate(Class<? extends DataCacheItem> item, List<String> updates) {
    Date itemUpdate = updated.getOrDefault(item.getName(), new Date(0));
    if (itemUpdate == null) {
      return false;
    }
    Stream.of(dependencies(item))
        .filter(i -> {
          final Date update = updateTime(i);
          return update != null && update.after(itemUpdate);
        })
        .forEach(updates::add);
    return updates.isEmpty();
  }
}
