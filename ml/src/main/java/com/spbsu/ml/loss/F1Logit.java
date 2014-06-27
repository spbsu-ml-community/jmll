package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 18:08
 */
public class F1Logit extends FBetaLogit {
  public F1Logit(Vec target) {
    super(target, 1.);
  }
}
