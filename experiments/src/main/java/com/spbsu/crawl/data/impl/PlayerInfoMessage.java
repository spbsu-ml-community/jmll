package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;
import org.jetbrains.annotations.NotNull;

import java.util.List;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class PlayerInfoMessage implements Message {
  public static int STAT_EMPTY_FIELD_VALUE = -1;

  @JsonProperty("title")
  private String title = null;

  @JsonProperty("hp")
  private int healthPoints = STAT_EMPTY_FIELD_VALUE;
  
  @JsonProperty("hp_max")
  private int maxHealthPoints = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("mp")
  private int manaPoints = STAT_EMPTY_FIELD_VALUE;
  
  @JsonProperty("mp_max")
  private int maxManPoints = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("gold")
  private int gold = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("time")
  private int time = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("turn")
  private int turn = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("depth")
  private String depth = null;
  
  @JsonProperty("str")
  private int strength = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("str_max")
  private int maxStrength = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("int")
  private int intelegence = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("int_max")
  private int maxIntelegence = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("dex")
  private int dexterity = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("dex_max")
  private int maxDexterity = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("ac")
  private int armorClass = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("sh")
  private int shieldClass = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("ev")
  private int evasion = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("xl")
  private int experienceLevel = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("progress")
  private int nextExpLevelProgress = STAT_EMPTY_FIELD_VALUE;

  @JsonProperty("status")
  private List<PlayerStatus> statuses = null;
  
  public String depth() {
    return depth;
  }

  @JsonProperty("pos")
  private PlayerPosition position = null;


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

  public int maxHealthPoints() {
    return maxHealthPoints;
  }

  public int maxManaPoints() {
    return maxManPoints;
  }

  public int strength() {
    return strength;
  }

  public int maxStrength() {
    return maxStrength;
  }

  public int intelegence() {
    return intelegence;
  }

  public int maxIntelegence() {
    return maxIntelegence;
  }

  public int dexterity() {
    return dexterity;
  }

  public int maxDexterity() {
    return maxDexterity;
  }

  public int armorClass() {
    return armorClass;
  }

  public int shieldClass() {
    return shieldClass;
  }

  public int evasion() {
    return evasion;
  }

  public int experienceLevel() {
    return experienceLevel;
  }

  public int nextExpLevelProgress() {
    return nextExpLevelProgress;
  }

  public List<PlayerStatus> statuses() {
    return statuses;
  }

  public PlayerPosition position() {
    return position;
  }

  public static class PlayerPosition {
    @JsonProperty("x")
    private int x;
    @JsonProperty("y")
    private int y;

    public int x() {
      return x;
    }

    public int y() {
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

  @JsonIgnoreProperties(ignoreUnknown = true)
  public static class PlayerStatus {
    @JsonProperty("light")
    private String text;
    @JsonProperty("text")
    private String lightText;
    @JsonProperty("col")
    private int colour;

    public String text() {
      return text;
    }

    public String lightText() {
      return lightText;
    }

    public int colour() {
      return colour;
    }
  }
}
