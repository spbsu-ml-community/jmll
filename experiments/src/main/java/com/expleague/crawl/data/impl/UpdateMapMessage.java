package com.expleague.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.expleague.crawl.data.Message;

import java.util.List;

public class UpdateMapMessage implements Message {

  @JsonProperty("clear")
  private boolean forceFullRedraw;

  @JsonProperty("player_on_level")
  private boolean isPlayerOnLevel;

  @JsonProperty("vgrdc")
  private CoordinateMessage cursorPosition; //i'm not sure, that is't cursor coordinates without some magic

  public boolean isForceFullRedraw() {
    return forceFullRedraw;
  }

  public boolean isPlayerOnLevel() {
    return isPlayerOnLevel;
  }

  public CoordinateMessage getCursorPosition() {
    return cursorPosition;
  }

  public List<UpdateMapCellMessage> getCells() {
    return Cells;
  }

  @JsonProperty("cells")
  private List<UpdateMapCellMessage> Cells;



}
