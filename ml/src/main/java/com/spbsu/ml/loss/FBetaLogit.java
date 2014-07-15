package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * User: starlight
 * Date: 10.09.13
 * Time: 18:08
 */
public class FBetaLogit extends LLLogit {
  private final PLogit precision;
  private final RLogit recall;
  private final double betta;

  public FBetaLogit(Vec target, DataSet<?> owner, double betta) {
    super(target, owner);
    this.betta = betta;
    this.precision = new PLogit(target, owner);
    this.recall = new RLogit(target, owner);
  }

  /**
   * @param point $y=-log(\frac{1}{p(x|c)} - 1)$ supposed to be there so that $p(x|c) = \frac{1}{1+e^{-x}}$
   */
  @Override
  public double value(Vec point) {
    double precisionValue = precision.value(point);
    double recallValue = recall.value(point);
    return (1 + betta * betta) * precisionValue * recallValue / (betta * betta * precisionValue + recallValue);
  }
}
