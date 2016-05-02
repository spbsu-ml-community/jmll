package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;
import org.jetbrains.annotations.NotNull;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class PlayerInfoMessage implements Message {
  @JsonProperty("title")
  private String title;

  @JsonProperty("hp")
  private int healthPoints;

  @JsonProperty("mp")
  private int manaPoints;

  @JsonProperty("gold")
  private int gold;

  @JsonProperty("time")
  private int time;

  @JsonProperty("turn")
  private int turn;

  @JsonProperty("depth")
  private String depth;

  public String depth() {
    return depth;
  }

  @JsonProperty("pos")
  private PlayerPosition position;


  public int healthPoints() {
    return healthPoints;
  }

  public int manaPoints() {
    return manaPoints;
  }

  public int gold() {
    return gold;
  }

  public int time() {
    return time;
  }

  public int turn() {
    return turn;
  }

  public String title() {
    return title;
  }

  @NotNull
  public PlayerPosition getPosition() {
    return position;
  }

  public static class PlayerPosition {
    @JsonProperty("x")
    private int x;
    @JsonProperty("y")
    private int y;

    public int getX() {
      return x;
    }

    public int getY() {
      return y;
    }
  }

  public static class InventoryThing {
    public static final InventoryThing NONE_ITEM = new InventoryThing(100, 0);

    @JsonProperty("base_type")
    private int baseType;

    @JsonProperty("quantity")
    private int quantity;

    @JsonProperty("sub_type")
    private int subType;

    @JsonProperty("flags")
    private int flags;

    public InventoryThing(int baseType, int quantity) {
      this.baseType = baseType;
      this.quantity = quantity;
    }
  }
}
