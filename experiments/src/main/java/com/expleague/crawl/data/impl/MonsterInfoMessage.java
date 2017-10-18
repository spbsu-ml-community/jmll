package com.expleague.crawl.data.impl;

import com.expleague.crawl.data.impl.system.EmptyFieldsDefault;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.expleague.crawl.data.Message;

/**
 * Created by noxoomo on 24/04/16.
 */

public class MonsterInfoMessage implements Message {
  private final boolean empty;
  private static MonsterInfoMessage emptyMessage = new MonsterInfoMessage(true);

  private MonsterInfoMessage(final boolean empty) {
    this.empty = empty;
  }

  public static MonsterInfoMessage emptyMessage() {
    return emptyMessage;
  }

  public MonsterInfoMessage() {
    this.empty = false;
  }

  public boolean isEmpty() {
    return empty;
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  public static class MonsterStats implements Message {
    @JsonProperty("avghp")
    private int averageHealthPoints =  EmptyFieldsDefault.emptyInt();
    @JsonProperty("no_exp")
    Boolean noExperience = null;

    public int averageHealthPoints() {
      return averageHealthPoints;
    }

    public void setAverageHealthPoints(final int averageHealthPoints) {
      this.averageHealthPoints = averageHealthPoints;
    }

    public void setNoExperience(final Boolean noExperience) {
      this.noExperience = noExperience;
    }

    public Boolean noExperience() {
      return noExperience;
    }

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

  public int monsterType() {
    return type;
  }

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

  public void setId(final int id) {
    this.id = id;
  }

  public void setName(final String name) {
    this.name = name;
  }

  public void setPlural(final String plural) {
    this.plural = plural;
  }

  public void setType(final int type) {
    this.type = type;
  }

  public MonsterInfoMessage setMonsterStats(final MonsterStats monsterStats) {
    this.monsterStats = monsterStats;
    return this;
  }

  public void setAttitude(final int attitude) {
    this.attitude = attitude;
  }

  public void setBaseType(final int baseType) {
    this.baseType = baseType;
  }

  public void setThreatLevel(final int threatLevel) {
    this.threatLevel = threatLevel;
  }
}


