package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

/**
 * Created by noxoomo on 24/04/16.
 */
public class MonsterInfoMessage implements Message {

  public static class MonsterStats implements Message {
    @JsonProperty("avghp")
    private int averageHealthPoints;
    @JsonProperty("no_exp")
    boolean noExperience;
  }

  @JsonProperty("id")
  private int id;

  @JsonProperty("name")
  private String name;

  @JsonProperty("plural")
  private String plural;

  @JsonProperty("type")
  private int type;

  @JsonProperty("typedata")
  private MonsterStats monsterStats;

  @JsonProperty("att")
  private int attitude;

  @JsonProperty("btype")
  private int baseType;

  @JsonProperty("threat")
  private int threatLevel;
}
