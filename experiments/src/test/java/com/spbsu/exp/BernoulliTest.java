package com.spbsu.exp;

import com.spbsu.bernulli.BernoulliMixture;
import com.spbsu.commons.random.FastRandom;
import junit.framework.TestCase;

import java.util.Arrays;

/**
 * bernoulli models experiments
 * User: noxoomo
 * Date: 09.03.2015
 */
public class BernoulliTest extends TestCase {
  FastRandom rand = new FastRandom();

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
      for (int i=1; i < q.length;++i) {
        cum[i] = cum[i-1] + q[i-1];
      }
      double[] sums = new double[N];
      double[] total = new double[N];
      double[] means = new double[N];
      Arrays.fill(total, n);
      for (int j=0; j < N;++j) {
//        final double coin = rand.nextDouble() ;
//        final int k = Math.abs(Arrays.binarySearch(cum,coin)) -2;
//        means[j] = theta[k];
        if (j > 0 && rand.nextDouble() > 0.3) {
          double m = means[rand.nextInt(j)];
          double s = Math.min(1-m,m) / 8;
          s *= (1-s);
          means[j] = m + (2*rand.nextDouble() * s - s);
        } else {
          means[j] = rand.nextDouble();
        }
//        means[j] = 0.1 + rand.nextDouble()*0.8;
        for (int s = 0; s < n;++s) {
          sums[j] += (rand.nextDouble() < means[j]) ? 1.0 : 0;
        }
      }
      return new Result(sums,total,means);
    }
  }


  public double l2(double[] a, double[] b) {
    assert(a.length == b.length);
    double sum = 0;
    for (int i=0; i < a.length;++i) {
      sum += (a[i] - b[i])*(a[i] - b[i]);
    }
    return sum;
  }
  public void testMixture() {
    final int k = 50;
    final int from = 500;
    final int to = 1000;
    final int step = 20;
    final int tries = 1000;
    BernoulliRand brand = new BernoulliRand();


    double[] theta = new double[k];
    double[] q = new double[k];
    for (int n = 50; n < 110; n += 10)
    for (int N = from; N < to; N += step) {
      double sumAvgMixture = 0;
      double sumAvgNaive = 0;
      for (int tr = 0; tr < tries;++tr) {
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
        for (int i=0; i < naive.length;++i) {
          naive[i] /= experiment.total[i];
        }
        sumAvgMixture += l2(means, experiment.means) / experiment.means.length;
        sumAvgNaive += l2(naive, experiment.means) / experiment.means.length;
      }
      System.out.println(N + "\t" + n +  "\t" + sumAvgMixture / tries + "\t" + sumAvgNaive / tries);
    }
  }

}
