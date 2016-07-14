package com.spbsu.crawl.data.impl.system;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class UpdateMenuMessage implements Message {
  @JsonProperty("title")
  String title; //it's json really
}
