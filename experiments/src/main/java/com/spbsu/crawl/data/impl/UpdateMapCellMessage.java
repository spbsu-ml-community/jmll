package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;


public class UpdateMapCellMessage implements Message {

  //this properties send if prop was updated
  //Almost all ints are uint32.
  @JsonProperty("x")
  private int x;

  @JsonProperty("y")
  private int y;

  @JsonProperty("f")
  private int dungeonFeatureType;

  @JsonProperty("mf")
  private int mapFeature;

  @JsonProperty("g")
  private String glyph;

  @JsonProperty("col")
  private int colour;

  @JsonProperty("t")
  private PackedCellMessage packedCell;

  @JsonProperty("mon")
  private MonsterInfoMessage monsterInfoMessage;

  public int getX() {
    return x;
  }

  public int getY() {
    return y;
  }

  public int getDungeonFeatureType() {
    return dungeonFeatureType;
  }

  public int getMapFeature() {
    return mapFeature;
  }

  public String getGlyph() {
    return glyph;
  }

  public int getColour() {
    return colour;
  }

  public MonsterInfoMessage getMonsterInfoMessage() {
    return monsterInfoMessage;
  }

  public PackedCellMessage getPackedCell() {
    return packedCell;
  }
}
