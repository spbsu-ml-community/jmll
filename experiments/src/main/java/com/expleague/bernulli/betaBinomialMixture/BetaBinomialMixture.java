package com.expleague.bernulli.betaBinomialMixture;

import com.expleague.bernulli.Mixture;
import com.expleague.bernulli.MixtureObservations;
import com.expleague.bernulli.Multinomial;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;

public class BetaBinomialMixture extends Mixture {
  public final double alphas[];
  public final double betas[];
  final Multinomial multinomialSampler;
  private final FastRandom rng = new FastRandom();
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
    multinomialSampler = new Multinomial(random, q);
    this.count = count;
  }


  public int sample() {
    int index = multinomialSampler.next();
    double q = rng.nextBeta(alphas[index], betas[index]);
    return rng.nextBinomial(count, q);
  }

  @Override
  public MixtureObservations<BetaBinomialMixture> sample(int n) {
    final int components[] = new int[n];
    final double[] thetas = new double[n];
    final int sums[] = new int[n];
    for (int i = 0; i < n; ++i) {
      components[i] = multinomialSampler.next();
      int index = components[i];
      thetas[i] = rng.nextBeta(alphas[index], betas[index]);
      sums[i] = rng.nextDouble() < thetas[i] ? 1 : 0;
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


