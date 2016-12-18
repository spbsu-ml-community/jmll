package com.spbsu.ml.data.softBorders;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by noxoomo on 12/12/2016.
 */
public class SimpleMCMCRunner<T> {
  private long burnInIterations = 100;
  private long runIterations = 1000000;
  private long estimationWindows = 100;
  private Sampler<T> sampler;
  private List<Estimator<T>> estimators = new ArrayList<>();

  public SimpleMCMCRunner<T> setSampler(final Sampler<T> sampler) {
    this.sampler = sampler;
    return this;
  }

  public SimpleMCMCRunner<T> setBurnInIterations(final int iters) {
    this.burnInIterations = iters;
    return this;
  }

  public SimpleMCMCRunner<T> setRunIterations(final int iters) {
    this.runIterations = iters;
    return this;
  }

  public SimpleMCMCRunner<T> setEstimationWindow(final int iters) {
    this.estimationWindows = iters;
    return this;
  }

  public SimpleMCMCRunner<T> addEsimator(Estimator<T> estimator) {
    this.estimators.add(estimator);
    return this;
  }

  public void run() {
    assert (sampler != null);
    assert (estimators.size() > 0);
    for (long i = 0; i < runIterations; ++i) {
      T sample = sampler.sample();
      if (i < burnInIterations || (i % estimationWindows != 0)) {
        continue;
      }
      estimators.forEach(estimator -> estimator.add(sample));
    }
  }
}
