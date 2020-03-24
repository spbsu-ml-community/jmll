package com.expleague.ml.embedding.lm;

import com.expleague.commons.math.vectors.Vec;

public interface LeftLanguageModel<T> {
  Vec advance(T word);

  Vec distribution();
  T best();
  double p(T next);
}
