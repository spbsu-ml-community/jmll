package com.expleague.ml.func;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.MathTools;

public class Gaussian1D extends AnalyticFunc.Stub {
  private static final double SQRT_TWO_PI = Math.sqrt(2 * Math.PI);
  private final double mu;
  private final double sigma;
  private final double sigma2sqr;
  private final double sigmaCube;

  public Gaussian1D(double mu, double sigmaSqr) {
    if (sigmaSqr <= 0) {
      throw new IllegalArgumentException("sigma must be positive");
    }

    this.mu = mu;
    this.sigma = Math.sqrt(sigmaSqr);
    sigma2sqr = 2 * sigmaSqr;
    sigmaCube = sigma * sigmaSqr;
  }

  public static double value(double x, double mu, double sigmaSqr) {
    return Math.exp(-MathTools.sqr(x - mu) / (2 * sigmaSqr))
        / SQRT_TWO_PI / Math.sqrt(sigmaSqr);
  }

  public static double gradient(double x, double mu, double sigmaSqr) {
    return - Math.exp(-MathTools.sqr(x - mu) / (2 * sigmaSqr))
        * (x - mu) / SQRT_TWO_PI / (sigmaSqr * Math.sqrt(sigmaSqr));
  }

  private double exp(double x) {
    return Math.exp(-MathTools.sqr(x - mu) / sigma2sqr);
  }

  @Override
  public double value(double x) {
    return exp(x) / SQRT_TWO_PI / sigma;
  }

  @Override
  public double gradient(double x) {
    return - exp(x) * (x - mu) / SQRT_TWO_PI / sigmaCube;
  }
}
