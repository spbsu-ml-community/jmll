package com.expleague.ml.distributions.samplers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.samplers.DistributionSampler;

import java.util.Random;
import java.util.function.DoubleConsumer;

public interface RandomVecSampler extends DistributionSampler<Vec> {

  double instance(final FastRandom random, int i);

  int dim();

  default Vec sampleTo(final FastRandom random,
                       final Vec dst) {
    for (int i = 0; i < dst.dim(); ++i) {
      dst.set(i, instance(random, i));
    }
    return dst;
  }

  default Vec sample(final FastRandom random) {
    Vec result = new ArrayVec(dim());
    return sampleTo(random, result);
  }

}
