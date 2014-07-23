package com.spbsu.ml.meta.items;

import com.spbsu.commons.math.vectors.impl.vectors.DVector;
import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 18.07.14
 * Time: 17:01
 */
public class ViewportAnswersWeighting implements DSItem {
  public String vpName;
  public DVector<String> answers;
  public String reqId;

  public ViewportAnswersWeighting(final String reqId, String vpName, final DVector<String> answers) {
    this.vpName = vpName;
    this.answers = answers;
    this.reqId = reqId;
  }

  public ViewportAnswersWeighting() {
  }

  @Override
  public String id() {
    return reqId;
  }
}
