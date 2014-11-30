package com.spbsu.ml.methods.greedyMergeOptimization;

/**
 * Created by noxoomo on 30/11/14.
 */
public interface RegularizedLoss<Model> {
  double target(Model model);

  double regularization(Model model);

  double score(Model model);

}

