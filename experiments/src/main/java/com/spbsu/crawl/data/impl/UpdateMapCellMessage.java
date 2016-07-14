package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;


public class UpdateMapCellMessage implements Message {

  //this properties send if prop was updated
  //Almost all ints are uint32.
  @JsonProperty("x")
  private int x = EmptyFieldsDefault.emptyInt();

  @JsonProperty("y")
  private int y = EmptyFieldsDefault.emptyInt();

  public void setPoint(final int x, final int y) {
    this.x = x;
    this.y = y;
  }

  @JsonProperty("f")
  private int dungeonFeatureType =  EmptyFieldsDefault.emptyInt();

  @JsonProperty("mf")
  private int mapFeature = EmptyFieldsDefault.emptyInt();

  @JsonProperty("g")
  private String glyph =  EmptyFieldsDefault.emptyValue();

  @JsonProperty("col")
  private int colour =  EmptyFieldsDefault.emptyInt();

  @JsonProperty("t")
  private PackedCellMessage packedCell =  EmptyFieldsDefault.emptyValue();

  @JsonProperty("mon")
  private MonsterInfoMessage monsterInfoMessage =  EmptyFieldsDefault.emptyValue();

  public int x() {
    return x;
  }

  public int y() {
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
