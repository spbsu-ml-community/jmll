package com.spbsu.bernulli;

/**
 * User: Noxoomo
 * Date: 20.03.15
 * Time: 16:38
 */
public abstract class EM<Result> {
  protected abstract void expectation();

  protected abstract void maximization();

  protected abstract boolean stop();

  public abstract Result model();

  protected abstract double likelihood();

  public final FittedModel<Result> fit() {
    return fit(false);
  }

  public final FittedModel<Result> fit(boolean correctnessTest) {
    if (!correctnessTest) {
      while (!stop()) {
        expectation();
        maximization();
      }
    } else {
      double ll = Double.NEGATIVE_INFINITY;
      while (!stop()) {
        expectation();
        maximization();
        double currentLL = likelihood();
        if (currentLL + 1e-2 < ll) {
          throw new RuntimeException("EM always increase likelihood");
        }
      }
    }
    return new FittedModel<>(likelihood(), model());
  }


}




