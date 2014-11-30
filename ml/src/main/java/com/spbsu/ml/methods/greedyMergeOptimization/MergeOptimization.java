package com.spbsu.ml.methods.greedyMergeOptimization;

/**
 * Created by noxoomo on 30/11/14.
 */

public interface MergeOptimization<Model> {
  Model merge(Model first, Model second);
}
