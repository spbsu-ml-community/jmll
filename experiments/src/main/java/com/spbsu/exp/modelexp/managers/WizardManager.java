package com.spbsu.exp.modelexp.managers;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.exp.modelexp.Experiment;
import com.spbsu.exp.modelexp.Query;

/**
* User: solar
* Date: 03.04.15
* Time: 17:51
*/
public class WizardManager implements Factory<Experiment[]> {
  private final FastRandom rng;
  private int expCount = 0;

  public WizardManager(FastRandom rng) {
    this.rng = rng;
  }

  @Override
  public Experiment[] create() {
    final double expMean = rng.nextGaussian() * 5 - 1;

    return new Experiment[]{new Experiment() {
      long count = 0;
      public String id = "" + expCount++;

      @Override
      public double work(Query q) {

        q.properties();
        count++;
        return rng.nextGaussian() + expMean;
      }

      @Override
      public boolean relevant(Query q) {
        return true;
      }

      @Override
      public double realScore() {
        return expMean;
      }

      @Override
      public String toString() {
        return "SERP: " + id + ", " + expMean;
      }
    }};
  }
}
