package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 18:08
 */
public class F1Logit extends FBetaLogit {
  public F1Logit(final Vec target, final DataSet<?> owner) {
    super(target, owner, 1.);
  }
}
