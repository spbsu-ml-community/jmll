package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 18:08
 */
public class FBetaLogit extends LLLogit {
  private final Vec target;
  private final double betta;

  public FBetaLogit(Vec target, double betta) {
    super(target);
    this.target = target;
    this.betta = betta;
  }

  /**
   * @param point $y=-log(\frac{1}{p(x|c)} - 1)$ supposed to be there so that $p(x|c) = \frac{1}{1+e^{-x}}$
   */
  @Override
  public double value(Vec point) {
    int truepositive = 0;
    int truenegative = 0;
    int falsepositive = 0;
    int falsenegative = 0;

    for (int i = 0; i < point.dim(); i++) {
      if (point.get(i) > 0 && target.get(i) > 0)
        truepositive++;
      else if (point.get(i) > 0 && target.get(i) <= 0)
        falsepositive++;
      else if (point.get(i) <= 0 && target.get(i) > 0)
        falsenegative++;
      else if (point.get(i) <= 0 && target.get(i) <= 0)
        truenegative++;
    }
    double precision = truepositive/(0. + truepositive + falsepositive);
    double recall = truepositive/(0. + truepositive + falsenegative);

    return (1 + betta*betta) * precision * recall/(betta * betta * precision + recall);
  }
}
