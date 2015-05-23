package com.spbsu.bernulli.naiveMixture;

import com.spbsu.bernulli.Mixture;
import com.spbsu.bernulli.MixtureObservations;
import com.spbsu.bernulli.Multinomial;
import com.spbsu.commons.random.FastRandom;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

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


  public double[] estimate(int[] sums) {
    final double[] result = new double[sums.length];
    final double[] logtheta  =new double[means.length];
    final double[] logntheta  =new double[means.length];
    final double[] logq  =new double[means.length];

    for (int i = 0; i < means.length; ++i) {
      logtheta[i] = FastMath.log(means[i]);
      logntheta[i] = FastMath.log(1 - means[i]);
      logq[i] = FastMath.log(q[i]);
    }
    for (int j = 0; j < sums.length; ++j) {
      double theta = 0;
      final double m = sums[j];
      final double n = count;
      double denum = 0;
      for (int i=0; i < means.length;++i) {
        double tmp = m != 0 ? m * logtheta[i] : 0;
        tmp += (n - m) != 0 ? (n - m) * logntheta[i] : 0;
        tmp += logq[i];
        final double p = FastMath.exp(tmp);
        denum += p;
        theta += p * means[i];
      }
      theta /= denum;
      result[j] = theta;
    }

    return result;
  }


}
