package com.spbsu.exp.dl.dnn.rectifiers;

import com.spbsu.commons.math.vectors.Mx;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 *
 * @author ksenon
 */
public interface Rectifier {

  void value(final @NotNull Mx x, final @NotNull Mx y);

  void grad(final @NotNull Mx x, final @NotNull Mx y);

}
