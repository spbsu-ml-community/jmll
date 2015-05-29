package com.spbsu.exp.dl.dnn.rectifiers;

import com.spbsu.commons.math.vectors.Mx;
import org.jetbrains.annotations.NotNull;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class BipolarSigmoid implements Rectifier {

  @Override
  public void value(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < x.dim(); i++) {
      y.set(i, (1.f - Math.exp(-x.get(i))) / (1.f + Math.exp(-x.get(i))));
    }
  }

  @Override
  public void grad(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < y.dim(); i++) {
      y.set(i, 2 * (Math.exp(x.get(i)) / Math.pow(Math.exp(x.get(i)) + 1., 2)));
    }
  }

}
