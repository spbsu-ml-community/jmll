package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

/**
 * Created by noxoomo on 24/04/16.
 */
public class MonsterInfoMessage implements Message {

  @JsonIgnoreProperties(ignoreUnknown = true)
  public static class MonsterStats implements Message {
    @JsonProperty("avghp")
    private int averageHealthPoints =  EmptyFieldsDefault.emptyInt();
    @JsonProperty("no_exp")
    boolean noExperience;

    boolean isUpdated() {
      return EmptyFieldsDefault.isEmpty(averageHealthPoints);
    }
  }

  @JsonProperty("id")
  private int id = EmptyFieldsDefault.emptyInt();

  @JsonProperty("name")
  private String name = EmptyFieldsDefault.emptyValue();

  @JsonProperty("plural")
  private String plural = EmptyFieldsDefault.emptyValue();

  @JsonProperty("type")
  private int type = EmptyFieldsDefault.emptyInt();

  @JsonProperty("typedata")
  private MonsterStats monsterStats = EmptyFieldsDefault.emptyValue();

  @JsonProperty("att")
  private int attitude = EmptyFieldsDefault.emptyInt();

  @JsonProperty("btype")
  private int baseType = EmptyFieldsDefault.emptyInt();

  @JsonProperty("threat")
  private int threatLevel = EmptyFieldsDefault.emptyInt();

  public int id() {
    return id;
  }

  public String name() {
    return name;
  }

  public String plural() {
    return plural;
  }

  public MonsterStats monsterStats() {
    return monsterStats;
  }

  public int attitude() {
    return attitude;
  }

  public int baseType() {
    return baseType;
  }

  public int threatLevel() {
    return threatLevel;
  }
}
