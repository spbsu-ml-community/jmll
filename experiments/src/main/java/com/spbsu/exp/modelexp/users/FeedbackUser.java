package com.spbsu.exp.modelexp.users;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.exp.modelexp.Query;
import com.spbsu.exp.modelexp.User;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.Arrays;

/**
* User: solar
* Date: 07.04.15
* Time: 14:43
*/
public class FeedbackUser extends UniformUser {
  private double lambda;

  public FeedbackUser(FastRandom rng, double lambda) {
    super(rng, lambda);
    this.lambda = lambda;
  }

  @Override
  public void feedback(double score) {
    if (rng.nextDouble() > Math.pow(0.99, 1/lambda)) // cookie dead
      lambda = 0;
    lambda *= Math.exp(0.0001 * (score - 1));
  }

  @Override
  public double activity() {
    return lambda;
  }
}
