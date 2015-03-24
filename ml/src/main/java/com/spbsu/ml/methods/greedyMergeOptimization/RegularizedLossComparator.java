package com.spbsu.ml.methods.greedyMergeOptimization;

import java.util.Comparator;

/**
 * Created by noxoomo on 30/11/14.
 */
public class RegularizedLossComparator<Model, Loss extends RegularizedLoss<Model>> implements ModelComparators<Model> {
  Loss loss;

  public RegularizedLossComparator(final Loss loss) {
    this.loss = loss;
  }

  @Override
  public Comparator<Model> regularizationComparator() {
    return new Comparator<Model>() {
      @Override
      public int compare(final Model left, final Model right) {
        return Double.compare(loss.regularization(left), loss.regularization(right));
      }
    };
  }

  @Override
  public Comparator<Model> targetComparator() {
    return new Comparator<Model>() {
      @Override
      public int compare(final Model left, final Model right) {
        return Double.compare(loss.target(left), loss.target(right));
      }
    };
  }

  @Override
  public Comparator<Model> scoreComparator() {
    return new Comparator<Model>() {
      @Override
      public int compare(final Model left, final Model right) {
        return Double.compare(loss.score(left), loss.score(right));
      }
    };
  }
}
