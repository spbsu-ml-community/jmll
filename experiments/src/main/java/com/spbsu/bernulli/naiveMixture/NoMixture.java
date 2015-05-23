package com.spbsu.bernulli.naiveMixture;

import com.spbsu.bernulli.Mixture;
import com.spbsu.bernulli.MixtureObservations;
import com.spbsu.commons.random.FastRandom;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * Created by noxoomo on 27/03/15.
 */
public class NoMixture extends Mixture {
  private final  RandomGenerator apacheRand;
  final int count;
  public NoMixture(int count, final FastRandom rand) {
    super(null,rand);
    this.count = count;
    this.apacheRand = new MersenneTwister(rand.nextLong());
  }


  public MixtureObservations<NoMixture> sample(int n) {
    int[] components = new int[n];
    double[] means = new double[n];
    int[] sums = new int[n];
    for (int i=0; i < components.length;++i) {
      means[i] = random.nextDouble();
      final BinomialDistribution dist = new BinomialDistribution(apacheRand,count,means[i]);
      sums[i] = dist.sample();
    }
    return new MixtureObservations<>(this,components,means,sums,count);
  }

}
