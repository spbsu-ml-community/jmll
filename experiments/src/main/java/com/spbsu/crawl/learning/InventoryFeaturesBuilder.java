package com.spbsu.crawl.learning;

import com.spbsu.crawl.bl.crawlSystemView.InventoryView;
import com.spbsu.crawl.bl.events.InventoryListener;
import com.spbsu.crawl.learning.features.Feature;
import com.spbsu.crawl.learning.features.NumericalFeature;

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * Created by noxoomo on 16/07/16.
 */
public class InventoryFeaturesBuilder implements InventoryListener, FeaturesBuilder {
  private final static String featureName = "Food";

  private final static String[] foodSubstrings = {"pizza", "bread", "meat", "beef"};
  private int[] counts = new int[InventoryView.INVENTORY_SIZE];
  private boolean[] isFood = new boolean[InventoryView.INVENTORY_SIZE];
  private int totalFoodCount = 0;

  private boolean isFood(final InventoryView.Item item) {
    return Arrays.stream(foodSubstrings).anyMatch(item.name()::contains);
  }

  @Override
  public void putItem(final int slot, final InventoryView.Item item, final int count) {
    counts[slot] = count;
    if (isFood(item)) {
      isFood[slot] = true;
      totalFoodCount += count;
    }
  }

  @Override
  public void add(final int slot, final int count) {
    counts[slot] += count;
    if (isFood[slot]) {
      totalFoodCount += count;
    }
  }

  @Override
  public void remove(final int slot, final int count) {
    add(slot, -count);
  }

  public int totalFoodCount() {
    return totalFoodCount;
  }

  @Override
  public Stream<Feature> tickFeatures() {
    return Stream.of(new NumericalFeature(totalFoodCount(), featureName));
  }
}
