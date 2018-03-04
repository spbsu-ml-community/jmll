package com.expleague.ml.distributions;

import com.expleague.commons.seq.Seq;

public interface RandomSeqBuilder<T, U extends Distribution<T>> {

  RandomSeqBuilder<T, U> add(final U distribution);
  default RandomSeqBuilder<T, U> add(final Seq<U> seq) {
    RandomSeqBuilder<T, U> result = this;
    for (int i = 0; i < seq.length(); ++i) {
      result = add(seq.at(i));
    }
    return result;
  }

  RandomSeq<T> build();



}


