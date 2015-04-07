package com.spbsu.bernulli.naiveMixture;

import com.spbsu.bernulli.Mixture;
import com.spbsu.bernulli.MixtureObservations;
import com.spbsu.bernulli.Multinomial;
import com.spbsu.commons.random.FastRandom;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

/**
 * Created by noxoomo on 27/03/15.
 */
public class NaiveMixture  extends Mixture {
  private final  RandomGenerator apacheRand;
  private final Multinomial multinomial;
  private final double[] means;
  private final int count;
  public NaiveMixture(final double[] q,final int count, final FastRandom rand) {
    super(q,rand);
    this.apacheRand = new MersenneTwister(rand.nextLong());
    this.multinomial =   new Multinomial(rand, q);
    this.means = new double[q.length];
    for (int i=0; i < q.length;++i) {
      this.means[i] = rand.nextDouble();
    }
    this.count = count;
  }
  public NaiveMixture(final int k,final int count, final FastRandom rand) {
    super(k,rand);
    this.apacheRand = new MersenneTwister(rand.nextLong());
    this.multinomial =   new Multinomial(rand, q);
    this.means = new double[q.length];
    for (int i=0; i < q.length;++i) {
      this.means[i] = rand.nextDouble();
    }
    this.count = count;
  }

  public MixtureObservations<NaiveMixture> sample(int n) {
    int[] components = new int[n];
    double[] means = new double[n];
    int[] sums = new int[n];
    for (int i=0; i < components.length;++i) {
      final int component = multinomial.next();
      components[i] = component;
      means[i] = this.means[component];
      final BinomialDistribution dist = new BinomialDistribution(apacheRand,count,means[i]);
      sums[i] = dist.sample();
    }
    return new MixtureObservations<>(this,components,means,sums,count);
  }

}
