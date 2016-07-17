package com.spbsu.crawl.bl.events;

import com.spbsu.crawl.bl.crawlSystemView.InventoryView;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface InventoryListener extends SystemViewListener {
  void putItem(final int slot, final InventoryView.Item item, int count);

  void add(final int slot, final int count);

  void remove(final int slot, final int count);
}
