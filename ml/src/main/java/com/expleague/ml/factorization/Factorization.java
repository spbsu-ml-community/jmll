package com.expleague.ml.factorization;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.Pair;

/**
 * User: qdeee
 * Date: 12.01.15
 */
public interface Factorization {
  Pair<Vec, Vec> factorize(final Mx X);
}
