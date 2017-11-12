package com.expleague.ml.distributions;

import com.expleague.commons.func.Action;

public interface RandomVecBuilder<U extends RandomVariable<U>> {
  RandomVecBuilder<U> add(final U distribution);
  RandomVec<U> build();
}
