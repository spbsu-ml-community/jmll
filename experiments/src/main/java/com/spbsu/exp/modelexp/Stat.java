package com.spbsu.exp.modelexp;

/**
 * User: solar
 * Date: 03.04.15
 * Time: 13:26
 */
public class Stat {
  double sum = 0;
  double sum2 = 0;
  int power = 0;

  public Stat update(double score) {
    power++;
    sum += score;
    sum2 += score * score;
    return this;
  }

  public Verdict status() {
    if (power < 3)
      return Verdict.INSIGNIFICANT;
    final double mean = sum/power;
    final double D = (1./(power - 1.)) * (sum2 - sum * mean);
    final double var = Math.sqrt(D);
    if (Math.abs(mean) < var * 3)
      return Verdict.INSIGNIFICANT;

    return mean > 0 ? Verdict.GOOD : Verdict.BAD;
  }


  public enum Verdict {
    GOOD, BAD, INSIGNIFICANT
  }
}
