package com.expleague.exp.modelexp.users;

import com.spbsu.commons.random.FastRandom;

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
