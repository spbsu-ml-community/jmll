package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.bl.events.InventoryListener;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

/**
 * Created by noxoomo on 16/07/16.
 */
public class InventoryView extends Subscribable.Stub<InventoryListener> implements Subscribable<InventoryListener> {
  public final static int INVENTORY_SIZE = 52;
  private final Item[] inventory = new Item[INVENTORY_SIZE];
  private final int[] counts = new int[INVENTORY_SIZE];
  private final Updater updater = new Updater();


  public Updater updater() {
    return updater;
  }

  class Updater {

    private void emptySlot(final int id) {
      final int cnt = counts[id];
      counts[id] = 0;
      inventory[id] = null;
      listeners().forEach(inventoryListener -> inventoryListener.remove(id, cnt));
    }

    private void updateItemCount(final int id,
                                 final int diff) {
      if (diff > 0) {
        counts[id] += diff;
        if (diff > 0) {
          listeners().forEach(inventoryListener -> inventoryListener.add(id, diff));
        } else {
          listeners().forEach(inventoryListener -> inventoryListener.remove(id, -diff));
        }
      }
    }

    private boolean isEmptySlot(final int id) {
      return counts[id] == 0;
    }

    private boolean notEmptySlot(final int id) {
      return !isEmptySlot(id);
    }

    private void putItemInSlot(final int id, final Item item, int count) {
      if (inventory[id] != null || counts[id] != 0) {
        throw new RuntimeException("Error: slot is not empty");
      }
      inventory[id] = item;
      counts[id] = count;
      listeners().forEach(inventoryListener -> inventoryListener.putItem(id, item, count));
    }

    public void item(final int id,
                     final PlayerInfoMessage.InventoryThing itemMessage) {

      if (itemMessage.quantity() == 0) {
        if (notEmptySlot(id)) {
          emptySlot(id);
        }
        return;
      }

      final Item item = new Item(inventory[id]);
      updateItemInfo(item, itemMessage);

      if (isSameItem(item, id)) {
        final int diff = itemMessage.quantity() - counts[id];
        updateItemCount(id, diff);
      } else {
        emptySlot(id);
        putItemInSlot(id, item, itemMessage.quantity());
      }
    }

    private void updateItemInfo(final Item item,
                                final PlayerInfoMessage.InventoryThing itemMessage) {
      if (EmptyFieldsDefault.notEmpty(itemMessage.baseType())) {
        item.type = itemMessage.baseType();
      }
      if (EmptyFieldsDefault.notEmpty(itemMessage.subType())) {
        item.subtype = itemMessage.subType();
      }
      if (EmptyFieldsDefault.notEmpty(itemMessage.name())) {
        item.name = itemMessage.name();
      }
    }


    private boolean isSameItem(Item item, int id) {
      return counts[id] > 0 && inventory[id].equals(item);
    }
  }

  public static class Item {
    private int type;
    private int subtype;
    private String name;

    public Item(Item item) {
      if (item != null) {
        type = item.type;
        subtype = item.subtype;
        name = item.name;
      }
    }

    public int type() {
      return type;
    }

    public int subtype() {
      return subtype;
    }

    public String name() {
      return name == null ? "Unknown item" : name;
    }

    public Item(PlayerInfoMessage.InventoryThing item) {
      type = item.baseType();
      name = item.name();
      subtype = item.subType();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;

      Item item = (Item) o;

      if (type != item.type) return false;
      if (subtype != item.subtype) return false;
      return name != null ? name.equals(item.name) : item.name == null;

    }

    @Override
    public int hashCode() {
      int result = type;
      result = 31 * result + subtype;
      result = 31 * result + (name != null ? name.hashCode() : 0);
      return result;
    }
  }


}
