package com.spbsu.exp.dl.dnn.rectifiers;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.math.vectors.Mx;

/**
 * jmll
 *
 * @author ksenon
 */
public class Flat implements Rectifier {

  @Override
  public void value(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < x.dim(); i++) {
      y.set(i, x.get(i));
    }
  }

  @Override
  public void grad(final @NotNull Mx x, final @NotNull Mx y) {
    for (int i = 0; i < y.dim(); i++) {
      y.set(i, 1);
    }
  }

}
