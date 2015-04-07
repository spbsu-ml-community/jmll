package com.spbsu.bernulli.naiveMixture;

import com.spbsu.bernulli.EM;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;

import java.util.Arrays;

/**
 * Created by noxoomo on 09/03/15.
 * <p/>
 * Mixture of bernoulli coins
 * let's q = (q_0,…,q_k) be some distribution (\sum q_i = 1) on (\theta_1, …, \theta_k)
 * we observe sequence of bernoulli events:
 * 1) choose \mu_i ~ q, (\mu_i = \theta_j for some j)
 * 2) toss n_i times a "coin" with parameter \mu_i
 * <p/>
 * Task: estimate \mu_i for every i
 * Subtask: estimate q_i and \theta_i
 */
public class BernoulliMixtureEM extends EM<NaiveMixture> {

  public double[] estimate(boolean needFit) {
    if (needFit) {
      fit();
    }
    final double p[] = new double[sums.length];
    expectation();
    for (int j = 0; j < sums.length; ++j) {
      double prob = 0;
      for (int i = 0; i < k; ++i) {
        prob += dummy.get(i, j) * theta[i];
      }
      p[j] = prob;
    }
    return p;
  }

  final int[] sums;
  final int total;
  final int k;
  final Mx dummy;
  final double[] cache;

  final double[] q;
  final double[] logq;
  final double[] theta;
  final double[] logtheta;
  final double[] logntheta;
  final FastRandom rand;

  public BernoulliMixtureEM(int[] sums, int total, int k, FastRandom rand) {
    this.sums = sums;
    this.total = total;
    this.k = k;
    this.dummy = new VecBasedMx(k, sums.length);
    cache = new double[k];
    q = new double[k];
    logq = new double[k];
    theta = new double[k];
    logtheta = new double[k];
    logntheta = new double[k];
    this.rand = rand;
    init();
  }

  private void init() {
    double totalWeight = 0;
    for (int i = 0; i < q.length; ++i) {
      q[i] = rand.nextDouble();
      totalWeight += q[i];
    }
    for (int i = 0; i < q.length; ++i) {
      q[i] /= totalWeight;
      logq[i] = Math.log(q[i]);
    }
    for (int i = 0; i < theta.length; ++i) {
      theta[i] = rand.nextDouble();

    }
    for (int i = 0; i < theta.length; ++i) {
      logtheta[i] = Math.log(theta[i]);
      logntheta[i] = Math.log(1 - theta[i]);
    }
    Arrays.sort(theta);
  }


  @Override
  protected void expectation() {
    final double n = total;
    for (int j = 0; j < sums.length; ++j) {
      final double m = sums[j];
      double denum = 0;
      final int length = (k / 4) * 4;
      for (int i = 0; i < length; i += 4) {
        final double llt0;
        final double llt1;
        final double llt2;
        final double llt3;
        if (m != 0) {
          llt0 = logtheta[i];
          llt1 = logtheta[i + 1];
          llt2 = logtheta[i + 2];
          llt3 = logtheta[i + 3];
        } else {
          llt0 = 0;
          llt1 = 0;
          llt2 = 0;
          llt3 = 0;
        }
        final double rlt0;
        final double rlt1;
        final double rlt2;
        final double rlt3;
        if (m != n) {
          rlt0 = logntheta[i];
          rlt1 = logntheta[i + 1];
          rlt2 = logntheta[i + 2];
          rlt3 = logntheta[i + 3];
        } else {
          rlt0 = 0;
          rlt1 = 0;
          rlt2 = 0;
          rlt3 = 0;
        }

        final double lq0 = logq[i];
        final double lq1 = logq[i + 1];
        final double lq2 = logq[i + 2];
        final double lq3 = logq[i + 3];

        final double tmp0 = m * llt0 + (n - m) * rlt0 + lq0;
        final double tmp1 = m * llt1 + (n - m) * rlt1 + lq1;
        final double tmp2 = m * llt2 + (n - m) * rlt2 + lq2;
        final double tmp3 = m * llt3 + (n - m) * rlt3 + lq3;
        cache[i] = Math.exp(tmp0);
        cache[i + 1] = Math.exp(tmp1);
        cache[i + 2] = Math.exp(tmp2);
        cache[i + 3] = Math.exp(tmp3);
        denum += (cache[i] + cache[i + 1] + cache[i + 2] + cache[i + 3]);
      }

      for (int i = length; i < k; ++i) {
        double tmp = m != 0 ? m * logtheta[i] : 0;
        tmp += (n - m) != 0 ? (n - m) * logntheta[i] : 0;
        tmp += logq[i];
        cache[i] = Math.exp(tmp);
        denum += cache[i];
      }

      for (int i = 0; i < k; ++i) {
        dummy.set(i, j, cache[i] != 0 && denum != 0 ? cache[i] / denum : 0);
      }

//      //test
//      {
//        double totalWeight = 0;
//        for (int i = 0; i < k; ++i) {
//          totalWeight += dummy.get(i, j);
//        }
//        if (Math.abs(totalWeight - 1.0) > 1e-6) {
//          System.err.println("Error: probs should sum to one");
//        }
//      }
    }
  }


  @Override
  protected double likelihood() {
    double ll = 0;
    final double n = total;
    for (int j = 0; j < sums.length; ++j) {
      final double m = sums[j];
      double p = 0;

      //hotspot generates sse instructions
      final int length = (k / 4) * 4;
      for (int i = 0; i < length; i += 4) {
        final double llt0;
        final double llt1;
        final double llt2;
        final double llt3;
        if (m != 0) {
          llt0 = logtheta[i];
          llt1 = logtheta[i + 1];
          llt2 = logtheta[i + 2];
          llt3 = logtheta[i + 3];
        } else {
          llt0 = 0;
          llt1 = 0;
          llt2 = 0;
          llt3 = 0;
        }
        final double rlt0;
        final double rlt1;
        final double rlt2;
        final double rlt3;
        if (m != n) {
          rlt0 = logntheta[i];
          rlt1 = logntheta[i + 1];
          rlt2 = logntheta[i + 2];
          rlt3 = logntheta[i + 3];
        } else {
          rlt0 = 0;
          rlt1 = 0;
          rlt2 = 0;
          rlt3 = 0;
        }

        final double lq0 = logq[i];
        final double lq1 = logq[i + 1];
        final double lq2 = logq[i + 2];
        final double lq3 = logq[i + 3];

        final double tmp0 = m * llt0 + (n - m) * rlt0 + lq0;
        final double tmp1 = m * llt1 + (n - m) * rlt1 + lq1;
        final double tmp2 = m * llt2 + (n - m) * rlt2 + lq2;
        final double tmp3 = m * llt3 + (n - m) * rlt3 + lq3;

        final double p0 = Math.exp(tmp0);
        final double p1 = Math.exp(tmp1);
        final double p2 = Math.exp(tmp2);
        final double p3 = Math.exp(tmp3);
        final double p02 = p0 + p2;
        final double p13 = p1 + p3;
        p += p02 + p13;
      }
      for (int i = length; i < k; ++i) {
        double tmp = m != 0 ? m * logtheta[i] : 0;
        tmp += (n - m) != 0 ? (n - m) * logntheta[i] : 0;
        tmp += logq[i];
        p += Math.exp(tmp);
      }
      ll += Math.log(p);
    }
    return ll;
  }

  @Override
  protected void maximization() {
    for (int i = 0; i < k; ++i) {
      double M = 0;
      double N = 0;
      double p = 0;
      int length = (sums.length / 4) * 4;
      for (int j = 0; j < length; j += 4) {
        final double prob0 = dummy.get(i, j);
        final double prob1 = dummy.get(i, j + 1);
        final double prob2 = dummy.get(i, j + 2);
        final double prob3 = dummy.get(i, j + 3);
        final double s0 = sums[j];
        final double s1 = sums[j + 1];
        final double s2 = sums[j + 2];
        final double s3 = sums[j + 3];
        final double m0 = prob0 * s0 + prob2 * s2;
        final double n0 = prob0 * total + prob2 * total;
        final double p0 = prob0 + prob2;
        final double m1 = prob1 * s1 + prob3 * s3;
        final double n1 = prob1 * total + prob3 * total;
        final double p1 = prob1 + prob3;
        M += m0 + m1;
        N += n0 + n1;
        p += p0 + p1;
      }

      for (int j = length; j < sums.length; ++j) {
        final double prob = dummy.get(i, j);
        M += prob * sums[j];
        N += prob * total;
        p += prob;
      }

      theta[i] = M / N;
      logtheta[i] = Math.log(theta[i]);
      logntheta[i] = Math.log(1-theta[i]);
      q[i] = p / sums.length;
      logq[i] = Math.log(q[i]);
    }
    //test
//    {
//      double totalWeight = 0;
//      for (int i = 0; i < q.length; ++i) {
//        totalWeight += q[i];
//      }
//      if (Math.abs(totalWeight - 1.0) > 1e-6) {
//        System.err.println("Error: probs should sum to one");
//      }
//    }
  }

  int count = 100;
  double oldLikelihood = Double.NEGATIVE_INFINITY;


  @Override
  protected boolean stop() {
    if (count % 10 == 0) {
      final double currentLL = likelihood();
      if (oldLikelihood + 1e-5 >= currentLL) {
        return true;
      }
      oldLikelihood = currentLL;
    }
    return --count <= 0;
  }

  @Override
  public NaiveMixture model() {
    return new NaiveMixture(theta, total, rand);
  }
}

