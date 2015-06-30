package com.spbsu.ml.func.generic;

/**
 * a * x^b
 * User: solar
 * Date: 29.06.15
 * Time: 16:48
 */
public class Pow extends ElementaryFunc {
  private final double a;
  private final double b;

  public Pow(double a, double b) {
    this.a = a;
    this.b = b;
  }

  @Override
  public double value(double x) {
    return a * Math.pow(x, b);
  }

  @Override
  public ElementaryFunc gradient() {
    if (b == 1)
      return new Const(a);
    return new Pow(a * b, b - 1);
  }
}
