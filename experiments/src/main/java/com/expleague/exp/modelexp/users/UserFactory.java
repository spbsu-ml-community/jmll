package com.expleague.exp.modelexp.users;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.exp.modelexp.User;

/**
* User: solar
* Date: 03.04.15
* Time: 15:56
*/
public class UserFactory implements Factory<User> {
  private final FastRandom rng;

  public UserFactory(FastRandom rng) {
    this.rng = rng;
  }

  @Override
  public User create() {
//    return new UniformUser(rng, rng.nextPoisson(10));
    return new FeedbackUser(rng, rng.nextPoisson(10));
  }

}
