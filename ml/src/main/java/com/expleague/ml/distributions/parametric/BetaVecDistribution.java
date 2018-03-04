package com.expleague.ml.distributions.parametric;

import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.RandomVec;

/**
 * Created by noxoomo on 22/10/2017.
 */

public interface BetaVecDistribution extends RandomVec {

  BetaDistribution at(final int idx);

  double alpha(final int idx);

  double beta(final int idx);



}
