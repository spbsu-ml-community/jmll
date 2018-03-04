package com.expleague.ml.distributions;

import com.expleague.commons.seq.Seq;

public interface RandomSeq<T> extends Distribution<Seq<T>> {

  Distribution<T> at(final int idx);

  int length();
}




