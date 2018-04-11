package com.expleague.ml.func.generic;

import org.jetbrains.annotations.NotNull;

/**
* User: solar
* Date: 26.05.15
* Time: 11:45
*/
public class Log extends ElementaryFunc {
  private final double a, b;

  public Log(double a, double b) {
    this.a = a;
    this.b = b;
  }

  @NotNull
  @Override
  public ElementaryFunc gradient() {
    return new Pow(a, -1);
  }

  @Override
  public double value(double x) {
    return Math.log(a * x + b);
  }
}
