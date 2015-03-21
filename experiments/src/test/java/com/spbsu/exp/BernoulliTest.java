package com.spbsu.exp;

import com.spbsu.bernulli.*;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import junit.framework.TestCase;
import org.apache.commons.math3.special.Gamma;

import java.util.Arrays;

/**
 * bernoulli models experiments
 * User: noxoomo
 * Date: 09.03.2015
 */
public class BernoulliTest extends TestCase {
  FastRandom rand  = new FastRandom(42);


  class BernoulliRand {
    class Result {
      final double[] sums;
      final double[] total;
      final double[] means;

      Result(double[] sums, double[] total, double[] means) {
        this.sums = sums;
        this.total = total;
        this.means = means;
      }
    }

    Result simulate(double[] theta, double[] q, int N, int n) {
      double[] cum = new double[q.length];
      cum[0] = 0;
      for (int i = 1; i < q.length; ++i) {
        cum[i] = cum[i - 1] + q[i - 1];
      }
      double[] sums = new double[N];
      double[] total = new double[N];
      double[] means = new double[N];
      Arrays.fill(total, n);
      for (int j = 0; j < N; ++j) {
//        final double coin = rand.nextDouble() ;
//        final int k = Math.abs(Arrays.binarySearch(cum,coin)) -2;
//        means[j] = theta[k];
        if (j > 0 && rand.nextDouble() > 0.3) {
          double m = means[rand.nextInt(j)];
          double s = Math.min(1 - m, m) / 8;
          s *= (1 - s);
          means[j] = m + (2 * rand.nextDouble() * s - s);
        } else {
          means[j] = rand.nextDouble();
        }
//        means[j] = 0.1 + rand.nextDouble()*0.8;
        for (int s = 0; s < n; ++s) {
          sums[j] += (rand.nextDouble() < means[j]) ? 1.0 : 0;
        }
      }
      return new Result(sums, total, means);
    }
  }


  public double l2(double[] a, double[] b) {
    assert (a.length == b.length);
    double sum = 0;
    for (int i = 0; i < a.length; ++i) {
      sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
  }


  public void testBetaBinomialMixture() {
    final int k = 5;
    final int n = 100;
    final int count = 500;
    int tries = 100;

    for (int i = 0; i < tries; ++i) {
      BetaBinomialMixture mixture = new BetaBinomialMixture(k, rand);
      BetaBinomialMixture.Observations observation = mixture.next(count, n);
      BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k, observation.sums, n, rand);
      FittedModel<BetaBinomialMixture> fittedModel = em.fit(true);
      System.out.println("Real model " + mixture);
      System.out.println("Fitted model " + fittedModel.model);
    }
  }

  public void testBetaBinomialMixtureEstimation() {
    final int k = 3;
    int tries = 100;

    final int from = 100;
    final int to = 100000;
    final int step = 1000;

    for (int n = 100; n < 10000; n += 1000)
      for (int N = from; N < to; N += step) {
        double sumAvgMixture = 0;
        double sumAvgNaive = 0;
        for (int tr = 0; tr < tries; ++tr) {
          BetaBinomialMixture mix = new BetaBinomialMixture(3, rand);
          final BetaBinomialMixture.Observations observations = mix.next(N, n);
          final int finaln = n;
          StochasticSearch<BetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<BetaBinomialMixtureEM>>() {
            @Override
            public Learner<BetaBinomialMixtureEM> create() {
              return new Learner<BetaBinomialMixtureEM>() {
                @Override
                public FittedModel<BetaBinomialMixtureEM> fit() {
                  BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k, observations.sums, finaln, rand);
                  FittedModel<BetaBinomialMixture> model = em.fit();
                  return new FittedModel<>(model.likelihood, em);
                }
              };
            }
          });
          BetaBinomialMixtureEM em = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);

          double[] means = em.estimate(false);
          double[] naive = new double[observations.sums.length];
          for (int i = 0; i < naive.length; ++i) {
            naive[i] = observations.sums[i] * 1.0 / n;
          }
          sumAvgMixture += l2(means, observations.thetas) / observations.thetas.length;
          sumAvgNaive += l2(naive, observations.thetas) / observations.thetas.length;
        }
        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + sumAvgNaive / tries);
      }
  }


  public void testSpecialFunctionCache() {
//    BetaFunctionsProportion prop = new  BetaFunctionsProportion(0.5,0.5,50);
    SpecialFunctionCache prop = new SpecialFunctionCache(1, 1, 10);
//    assertTrue((prop.calculate(10,50)-1.860013246596769 * 1e-13) < 1e-20);
//    prop.update(1,1);
    assertTrue(Math.abs(prop.calculate(2, 10) - 0.0020202) < 1e-7);
    assertTrue(Math.abs(prop.calculate(3, 10) - 0.000757576) < 1e-7);
    assertTrue(Math.abs(prop.calculate(4, 10) - 0.0004329) < 1e-7);
    assertTrue(Math.abs(prop.calculate(5, 10) - 0.00036075) < 1e-7);
    assertTrue(Math.abs(prop.calculate(9, 10) - 0.00909091) < 1e-7);
    assertTrue(Math.abs(prop.calculate(10, 10) - 0.0909091) < 1e-7);

    for (int i = 0; i < 10; ++i) {
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.Alpha, i) - Gamma.digamma(1 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.Beta, i) - Gamma.digamma(1 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.AlphaBeta, i) - Gamma.digamma(2 + i)) < 1e-12);
    }

    prop.update(102.5, 10.11);
//    prop = new  BetaFunctionsProportion(102.5,10.11,10);
    assertTrue(Math.abs(prop.calculate(5, 10) - 6.478713223568241e-6) < 1e-9);
    assertTrue(Math.abs(prop.calculate(8, 10) - 0.00369375) < 1e-8);

    for (int i = 0; i < 10; ++i) {
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.Alpha, i) - Gamma.digamma(102.5 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.Beta, i) - Gamma.digamma(10.11 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.AlphaBeta, i) - Gamma.digamma(102.5 + 10.11 + i)) < 1e-12);
    }

    for (int i = 0; i < 10; ++i) {
      assertTrue(Math.abs(prop.digamma1(SpecialFunctionCache.Type.Alpha, i) - Gamma.trigamma(102.5 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma1(SpecialFunctionCache.Type.Beta, i) - Gamma.trigamma(10.11 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma1(SpecialFunctionCache.Type.AlphaBeta, i) - Gamma.trigamma(102.5 + 10.11 + i)) < 1e-12);
    }

    prop = new SpecialFunctionCache(12.2, 55.1, 100);

    assertTrue(Math.abs(prop.calculate(30, 100) - 3.520721627628687e-28) < 1e-32);
    assertTrue(Math.abs(prop.calculate(60, 100) - 7.007620723590574e-37) < 1e-41);
    assertTrue(Math.abs(prop.calculate(10, 100) - 1.764175281317258e-15) < 1e-19);
    assertTrue(Math.abs(prop.calculate(88, 100) - 3.97033387162681e-36) < 1e-40);


    for (int i = 0; i < 100; ++i) {
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.Alpha, i) - Gamma.digamma(12.2 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.Beta, i) - Gamma.digamma(55.1 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(SpecialFunctionCache.Type.AlphaBeta, i) - Gamma.digamma(12.2 + 55.1 + i)) < 1e-12);
    }
  }

  public void testMixture() {
    final int k = 20;
    final int from = 100;
    final int to = 200;
    final int step = 50;
    final int tries = 1000;
    BernoulliRand brand = new BernoulliRand();


    double[] theta = new double[k];
    double[] q = new double[k];
    for (int n = 50; n < 70; n += 10)
      for (int N = from; N < to; N += step) {
        double sumAvgMixture = 0;
        double sumAvgNaive = 0;
        for (int tr = 0; tr < tries; ++tr) {
          {
            double totalWeight = 0;
            for (int i = 0; i < q.length; ++i) {
              q[i] = rand.nextDouble();
              totalWeight += q[i];
            }
            for (int i = 0; i < q.length; ++i) {
              q[i] /= totalWeight;
            }
          }
          {
            for (int i = 0; i < theta.length; ++i) {
              theta[i] = rand.nextDouble() / 4;
            }
          }
          BernoulliRand.Result experiment = brand.simulate(theta, q, N, n);
          BernoulliMixture em = new BernoulliMixture();
          double[] means = em.estimate(experiment.sums, experiment.total, k);
          double[] naive = experiment.sums.clone();
          for (int i = 0; i < naive.length; ++i) {
            naive[i] /= experiment.total[i];
          }
          sumAvgMixture += l2(means, experiment.means) / experiment.means.length;
          sumAvgNaive += l2(naive, experiment.means) / experiment.means.length;
        }
        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + sumAvgNaive / tries);
      }
  }


}
