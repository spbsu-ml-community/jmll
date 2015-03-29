package com.spbsu.bernulli.betaBinomialMixture;

import com.spbsu.bernulli.Mixture;
import com.spbsu.bernulli.MixtureObservations;
import com.spbsu.bernulli.Multinomial;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

public class BetaBinomialMixture extends Mixture {
  public final double alphas[];
  public final double betas[];
  final RandomGenerator apacheRandom;
  final BetaDistribution[] samplers;
  final Multinomial multinomialSampler;
  private int count;


  void setCount(int count) {
    this.count = count;
  }

  public BetaBinomialMixture(double[] alphas, double[] betas, double[] q, int count) {
    this(alphas, betas, q, count, new FastRandom());
  }

  public BetaBinomialMixture(double[] alphas, double[] betas, double[] q, int count, FastRandom rand) {
    super(q, rand);
    this.alphas = alphas;
    this.betas = betas;
    ArrayTools.parallelSort(q, alphas, betas);
    this.apacheRandom = new MersenneTwister(random.nextInt());
    this.samplers = new BetaDistribution[q.length];
    for (int i = 0; i < samplers.length; ++i) {
      samplers[i] = new BetaDistribution(apacheRandom, alphas[i], betas[i], BetaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }
    multinomialSampler = new Multinomial(random, q);
    this.count = count;
  }


  private int randomInitLimit = 100;

  public BetaBinomialMixture(int k, int count, FastRandom random) {
    super(k, random);
    double[] alphas = new double[q.length];
    double[] betas = new double[q.length];
    for (int i = 0; i < q.length; ++i) {
      alphas[i] = random.nextDouble()*randomInitLimit;
      betas[i] = random.nextDouble()*randomInitLimit;
    }
    this.alphas = alphas;
    this.betas = betas;
    this.apacheRandom = new MersenneTwister(random.nextInt());
    this.samplers = new BetaDistribution[q.length];
    for (int i = 0; i < samplers.length; ++i) {
      samplers[i] = new BetaDistribution(apacheRandom, alphas[i], betas[i], BetaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
    }
    multinomialSampler = new Multinomial(random, q);
    this.count = count;
  }


  public int sample() {
    BinomialDistribution sampler = new BinomialDistribution(apacheRandom, count, samplers[multinomialSampler.next()].sample());
    return sampler.sample();
  }

  @Override
  public MixtureObservations<BetaBinomialMixture> sample(int n) {
    final int components[] = new int[n];
    final double[] thetas = new double[n];
    final int sums[] = new int[n];
    for (int i = 0; i < n; ++i) {
      components[i] = multinomialSampler.next();
      thetas[i] = samplers[components[i]].sample();
      BinomialDistribution sampler = new BinomialDistribution(apacheRandom, count, thetas[i]);
      sums[i] = sampler.sample();
    }
    return new MixtureObservations<>(this, components, thetas, sums, count);
  }


  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(alphas.length).append(": ");
    for (int i = 0; i < alphas.length; ++i) {
      builder.append("(").append(q[i]).append(",").append(alphas[i]).append(",").append(betas[i])
              .append(",").append(alphas[i] / (alphas[i]+betas[i])).append(")");
    }
    return builder.toString();
  }
}


