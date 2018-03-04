package com.expleague.ml.distributions;

public interface RandomList<U extends RandomVariable> extends RandomVecBuilder<U> {
  U get(int idx);
  void set(int idx, final U u);

  RandomVecBuilder<U> add(final U dist);

}
