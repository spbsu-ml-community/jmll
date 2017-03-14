package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

import java.util.List;
import java.util.Map;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class PlayerInfoMessage implements Message {
  @JsonProperty("title")
  private String title = EmptyFieldsDefault.emptyValue();
  
  @JsonProperty("hp")
  private int healthPoints = EmptyFieldsDefault.emptyInt();
  
  @JsonProperty("hp_max")
  private int maxHealthPoints = EmptyFieldsDefault.emptyInt();

  @JsonProperty("mp")
  private int manaPoints = EmptyFieldsDefault.emptyInt();
  
  @JsonProperty("mp_max")
  private int maxManPoints = EmptyFieldsDefault.emptyInt();

  @JsonProperty("gold")
  private int gold = EmptyFieldsDefault.emptyInt();

  @JsonProperty("time")
  private int time = EmptyFieldsDefault.emptyInt();

  @JsonProperty("turn")
  private int turn = EmptyFieldsDefault.emptyInt();

  @JsonProperty("depth")
  private String depth = EmptyFieldsDefault.emptyValue();
  
  @JsonProperty("str")
  private int strength = EmptyFieldsDefault.emptyInt();

  @JsonProperty("str_max")
  private int maxStrength = EmptyFieldsDefault.emptyInt();

  @JsonProperty("int")
  private int intelligence = EmptyFieldsDefault.emptyInt();

  @JsonProperty("int_max")
  private int maxIntelligence = EmptyFieldsDefault.emptyInt();

  @JsonProperty("dex")
  private int dexterity = EmptyFieldsDefault.emptyInt();

  @JsonProperty("dex_max")
  private int maxDexterity = EmptyFieldsDefault.emptyInt();

  @JsonProperty("ac")
  private int armorClass = EmptyFieldsDefault.emptyInt();

  @JsonProperty("sh")
  private int shieldClass = EmptyFieldsDefault.emptyInt();

  @JsonProperty("ev")
  private int evasion = EmptyFieldsDefault.emptyInt();

  @JsonProperty("xl")
  private int experienceLevel = EmptyFieldsDefault.emptyInt();

  @JsonProperty("progress")
  private int nextExpLevelProgress = EmptyFieldsDefault.emptyInt();

  @JsonProperty("status")
  private List<PlayerStatus> statuses = EmptyFieldsDefault.emptyValue();
  
  public String depth() {
    return depth;
  }

  @JsonProperty("pos")
  private PlayerPosition position = EmptyFieldsDefault.emptyValue();

  public Map<String, InventoryThing> items() {
    return items;
  }

  @JsonProperty("inv")
  private Map<String, InventoryThing> items = EmptyFieldsDefault.emptyValue();


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

  public int intelligence() {
    return intelligence;
  }

  public int maxIntelligence() {
    return maxIntelligence;
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
    private int x = EmptyFieldsDefault.emptyInt();
    @JsonProperty("y")
    private int y = EmptyFieldsDefault.emptyInt();

    public int x() {
      return x;
    }

    public int y() {
      return y;
    }
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  public static class InventoryThing {
    @JsonProperty("base_type")
    private int baseType = EmptyFieldsDefault.emptyInt();

    @JsonProperty("quantity")
    private int quantity = EmptyFieldsDefault.emptyInt();

    @JsonProperty("sub_type")
    private int subType = EmptyFieldsDefault.emptyInt();

    @JsonProperty("flags")
    private int flags = EmptyFieldsDefault.emptyInt();

    @JsonProperty("plus")
    private int plus = EmptyFieldsDefault.emptyInt();

    @JsonProperty("plus2")
    private int plus2 = EmptyFieldsDefault.emptyInt();

    @JsonProperty("inscription")
    private String inscription = EmptyFieldsDefault.emptyValue();

    @JsonProperty("name")
    private String name = EmptyFieldsDefault.emptyValue();

    public int quantity() {
      return quantity;
    }

    public String name() {
      return name;
    }

    public int baseType() {
      return baseType;
    }

    public int subType() {
      return subType;
    }
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  public static class PlayerStatus {
    @JsonProperty("light")
    private String text = EmptyFieldsDefault.emptyValue();
    @JsonProperty("text")
    private String lightText = EmptyFieldsDefault.emptyValue();
    @JsonProperty("col")
    private int colour = EmptyFieldsDefault.emptyInt();

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
