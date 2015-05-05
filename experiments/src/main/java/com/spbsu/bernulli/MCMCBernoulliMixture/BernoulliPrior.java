package com.spbsu.bernulli.MCMCBernoulliMixture;

/**
 * Created by noxoomo on 26/03/15.
 */

public interface BernoulliPrior {
  public double likelihood(int sum, int total);
}
