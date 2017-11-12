package com.expleague.ml.randomnessAware;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.set.VecDataSet;

/**
 * Created by noxoomo on 22/10/2017.
 */
public interface RandomnessAwareTrans<U extends ProcessRandomnessPolicy> extends Trans {

  U activePolicy();

  RandomnessAwareTrans<U> changePolicy(final U policy);

  void setRandom(final FastRandom random);

  FastRandom random();

  public Mx transAll(final VecDataSet dataSet);

  public abstract class Stub<U extends ProcessRandomnessPolicy> extends Trans.Stub implements RandomnessAwareTrans<U> {
    private U policy;
    private FastRandom random = null;

    public Stub(final U policy) {
      this.policy = policy;
    }

    @Override
    public U activePolicy() {
      return policy;
    }

    @Override
    public void setRandom(final FastRandom fastRandom) {
      this.random = fastRandom;
    }

    public FastRandom random() {
      if (random == null) {
        throw new RuntimeException("set random first");
      }
      return random;
    }

    @Override
    public RandomnessAwareTrans<U> changePolicy(final U policy) {
      this.policy = policy;
      return this;
    };
  }
}



