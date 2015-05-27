package com.spbsu.exp.dl.dnn.rectifiers;

import com.spbsu.commons.math.vectors.Mx;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 *
 * @author ksenon
 */
public class Sigmoid implements Rectifier {

  @Override
  public void value(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < x.dim(); i++) {
      y.set(i, sigmoid(x.get(i)));
    }
  }

  @Override
  public void grad(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < y.dim(); i++) {
      y.set(i, sigmoid(x.get(i)) * (1 - sigmoid(x.get(i))));
    }
  }

  private double sigmoid(final double x) {
    return 1. / (1. + Math.exp(-x));
  }

}
