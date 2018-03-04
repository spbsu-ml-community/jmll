package com.expleague.ml.distributions;

public interface AdditiveFamilyDistribution<U extends Distribution<?>> {


  U add(final U other,
        double scale);

  default U add(final U other) {
    return add(other, 1.0);
  }


  U scale(double scale);

}


