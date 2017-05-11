package com.spbsu.ml.factorization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Pair;

/**
 * User: qdeee
 * Date: 12.01.15
 */
public interface Factorization {
  Pair<Vec, Vec> factorize(final Mx X);
}
