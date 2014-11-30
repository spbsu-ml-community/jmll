package com.spbsu.ml.methods.greedyMergeOptimization;

import java.util.Comparator;

/**
 * Created by noxoomo on 30/11/14.
 */
public interface ModelComparators<Model> {
  Comparator<Model> regularizationComparator();

  Comparator<Model> targetComparator();

  Comparator<Model> scoreComparator();

}

