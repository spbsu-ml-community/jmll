package com.spbsu.bernulli.MCMCBernoulliMixture;

/**
 * Created by noxoomo on 26/03/15.
 */

public interface BernoulliPrior {
  double likelihood(int sum, int total);
  default double estimate(int sum, int total) {
    return sum * 1.0  / total;
  }

}
