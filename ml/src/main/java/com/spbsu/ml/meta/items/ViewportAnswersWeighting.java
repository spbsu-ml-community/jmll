package com.spbsu.ml.meta.items;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 18.07.14
 * Time: 17:01
 */
@JsonIgnoreProperties(ignoreUnknown=true)
public class ViewportAnswersWeighting extends DSItem.Stub implements DSItem {
  public String reqId;
  public String vpName;

  public ViewportAnswersWeighting(final String reqId, final String vpName) {
    this.vpName = vpName;
    this.reqId = reqId;
  }

  public ViewportAnswersWeighting() {
  }

  @Override
  public String id() {
    return reqId;
  }
}
