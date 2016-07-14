package com.spbsu.crawl.learning.featureBuilders;

import gnu.trove.map.hash.TObjectIntHashMap;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by noxoomo on 14/07/16.
 */
public class CategoricalFeaturesMap {
  private TObjectIntHashMap<String> direct = new TObjectIntHashMap<>();
  private List<String> inverse = new ArrayList<>();

  private synchronized int addNewEntryAndReturn(String value) {
    if (direct.containsKey(value)) {
      return direct.get(value);
    }
    final int nextId = direct.size();
    direct.put(value, nextId);
    inverse.add(value);
    return nextId;
  }

  public int value(@NotNull String message) {
    if (direct.containsKey(message)) {
      return direct.get(message);
    } else {
      return addNewEntryAndReturn(message);
    }
  }

  public String value(int id) {
    if (id >= 0 && id < inverse.size()) {
      return inverse.get(id);
    }
    return null;
  }

}
