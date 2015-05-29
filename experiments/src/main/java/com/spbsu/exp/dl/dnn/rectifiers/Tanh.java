package com.spbsu.exp.dl.dnn.rectifiers;

import com.spbsu.commons.math.vectors.Mx;
import org.jetbrains.annotations.NotNull;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class Tanh implements Rectifier {

  @Override
  public void value(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < x.dim(); i++) {
      y.set(i, Math.tanh(x.get(i)));
    }
  }

  @Override
  public void grad(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < y.dim(); i++) {
      y.set(i, Math.pow(1.f / Math.cosh(x.get(i)), 2.));
    }
  }

}
