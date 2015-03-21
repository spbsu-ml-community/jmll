package com.spbsu.bernulli;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import static java.util.Arrays.sort;

public class BetaBinomialMixture {
  final double alphas[];
  final double betas[];
  final double q[]; //probs of i's mixture
  final FastRandom random;
  final RandomGenerator apacheRandom;
  final BetaDistribution[] samplers;
  final Multinomial multinomialSampler;


  public BetaBinomialMixture(double[] alphas, double[] betas, double[] q) {
    this(alphas, betas, q, new FastRandom());
  }

  public BetaBinomialMixture(double[] alphas, double[] betas, double[] q, FastRandom rand) {
    this.alphas = alphas;
    this.betas = betas;
    this.q = q;
    ArrayTools.parallelSort(q, alphas, betas);
    this.random = rand;
    this.apacheRandom = new MersenneTwister(random.nextInt());
    this.samplers = new BetaDistribution[q.length];
    for (int i = 0; i < samplers.length; ++i) {
      samplers[i] = new BetaDistribution(apacheRandom, alphas[i], betas[i], BetaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }
    multinomialSampler = new Multinomial(random, q);
  }


  private int randomInitLimit = 100;

  public BetaBinomialMixture(int k, FastRandom random) {
    double q[] = new double[k];
    double total = 0;
    for (int i = 0; i < q.length; ++i) {
      q[i] = random.nextDouble();
      total += q[i];
    }
    for (int i = 0; i < q.length; ++i) {
      q[i] /= total;
    }

    sort(q);
    double[] alphas = new double[q.length];
    double[] betas = new double[q.length];
    for (int i = 0; i < q.length; ++i) {
      alphas[i] = randomInitLimit * random.nextDouble();
      betas[i] = randomInitLimit * random.nextDouble();
    }
    this.alphas = alphas;
    this.betas = betas;
    this.q = q;
    this.random = random;
    this.apacheRandom = new MersenneTwister(random.nextInt());
    this.samplers = new BetaDistribution[q.length];
    for (int i = 0; i < samplers.length; ++i) {
      samplers[i] = new BetaDistribution(apacheRandom, alphas[i], betas[i], BetaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }
    multinomialSampler = new Multinomial(random, q);
  }


  public int next(int n) {
    BinomialDistribution sampler = new BinomialDistribution(apacheRandom, n, samplers[multinomialSampler.next()].sample());
    return sampler.sample();
  }

  public Observations next(int count, int n) {
    final int components[] = new int[count];
    final double[] thetas = new double[count];
    final int sums[] = new int[count];
    for (int i = 0; i < count; ++i) {
      components[i] = multinomialSampler.next();
      thetas[i] = samplers[components[i]].sample();
      BinomialDistribution sampler = new BinomialDistribution(apacheRandom, n, thetas[i]);
      sums[i] = sampler.sample();
    }
    return new Observations(this, components, thetas, sums, n);
  }

  public class Observations {
    public final BetaBinomialMixture owner;
    public final int components[];
    public final double[] thetas; //theta[i] ~ Beta(alpha[components[i]], beta[components[i]])
    public final int sums[]; //sums[i] = Sum Bernoulli(theta[i]
    public final int n;

    Observations(BetaBinomialMixture owner, int[] components, double[] thetas, int[] sums, int n) {
      this.owner = owner;
      this.components = components;
      this.thetas = thetas;
      this.sums = sums;
      this.n = n;
    }
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(alphas.length).append(": ");
    for (int i = 0; i < alphas.length; ++i) {
      builder.append("(").append(q[i]).append(",").append(alphas[i]).append(",").append(betas[i]).append(")");
    }
    return builder.toString();
  }
}


