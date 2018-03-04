package com.expleague.ml.distributions;

import com.expleague.commons.seq.Seq;

public interface RandomVecBuilder<U extends RandomVariable> extends RandomSeqBuilder<Double, U> {

  RandomVecBuilder<U> add(final U distribution);

  RandomVec build();


}
