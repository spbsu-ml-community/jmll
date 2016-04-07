package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

import java.util.List;

public class UpdateMapMessage implements Message {

  @JsonProperty("clear")
  private boolean forceFullRedraw;

  @JsonProperty("player_on_level")
  private boolean isPlayerOnLevel;

  @JsonProperty("vgrdc")
  private CoordinateMessage cursorPosition; //i'm not sure, that is't cursor coordinates without some magic


  @JsonProperty("cells")
  private List<UpdateMapCellMessage> Cells; //TODO: we'll it work like this???


}
