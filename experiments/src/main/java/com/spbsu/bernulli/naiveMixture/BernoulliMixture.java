package com.spbsu.bernulli.naiveMixture;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 09/03/15.
 *
 * Mixture of bernoulli coins
 * let's q = (q_0,…,q_k) be some distribution (\sum q_i = 1) on (\theta_1, …, \theta_k)
 * we observe sequence of bernoulli events:
 *    1) choose \mu_i ~ q, (\mu_i = \theta_j for some j)
 *    2) toss n_i times a "coin" with parameter \mu_i
 *
 * Task: estimate \mu_i for every i
 * Subtask: estimate q_i and \theta_i
 */
public class BernoulliMixture {

  // \hat{\theta} and \hat{q_i} by EM
  public Pair<double[],double[]> estimateModel(double[] sums, double[] total, int k) {
    EMBernoulli em = new EMBernoulli(sums,total,k);
    return em.fit();
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("EM thread", -1);


  //means of i-estimation
  public double[] estimate(final double[] sums,final double[] total,final int k) {
    final BestHolder<EMBernoulli> bestHolder = new BestHolder<>();
    final int tries = 2;
    final CountDownLatch latch = new CountDownLatch(tries);
    for (int i=0; i < tries; ++i) {
      exec.submit(new Runnable() {
        @Override
        public void run() {
          final EMBernoulli em = new EMBernoulli(sums, total, k);
          em.fit();
          bestHolder.update(em,em.likelihood());
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      //
    }

    final EMBernoulli em = bestHolder.getValue();
    final double p[] = new double[sums.length];
    em.expectation();
    for (int j=0; j < sums.length;++j) {
      double prob = 0;
      for (int i=0; i < k;++i) {
        prob += em.dummy.get(i,j) * em.theta[i];
      }
      p[j] = prob;
    }
    return p;
  }
}


class EMBernoulli {
  final double[] sums;
  final double[] total;
  final int k;
  final Mx dummy;
  final double[] cache;

  final double[] q;
  final double[] theta;

  public EMBernoulli(double[] sums, double[] total, int k) {
    this.sums = sums;
    this.total = total;
    this.k = k;
    this.dummy = new VecBasedMx(k, sums.length);
    cache = new double[k];
    q = new double[k];
    theta = new double[k];
  }


  void expectation() {
    for (int j=0; j < sums.length;++j) {
      final double m = sums[j];
      final double n = total[j];
      double denum = 0;

      final int length = (k / 4) * 4;

      for (int i=0; i < length;i+=4) {
        final double t0 = theta[i];
        final double t1 = theta[i+1];
        final double t2 = theta[i+2];
        final double t3 = theta[i+3];

        final double llt0;
        final double llt1;
        final double llt2;
        final double llt3;
        if (m !=0) {
          llt0 = Math.log(t0);
          llt1 = Math.log(t1);
          llt2 = Math.log(t2);
          llt3 = Math.log(t3);
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
        if (m!=n) {
          rlt0 = Math.log(1-t0);
          rlt1 = Math.log(1-t1);
          rlt2 = Math.log(1-t2);
          rlt3 = Math.log(1-t3);
        } else {
          rlt0 = 0;
          rlt1 = 0;
          rlt2 = 0;
          rlt3 = 0;
        }

        final double  lq0 = Math.log(q[i]);
        final double  lq1 = Math.log(q[i+1]);
        final double  lq2 = Math.log(q[i+2]);
        final double  lq3 = Math.log(q[i+3]);

        final double tmp0 = m * llt0 + (n-m) * rlt0 + lq0;
        final double tmp1 = m * llt1 + (n-m) * rlt1 + lq1;
        final double tmp2 = m * llt2 + (n-m) * rlt2 + lq2;
        final double tmp3 = m * llt3 + (n-m) * rlt3 + lq3;
        cache[i] = Math.exp(tmp0);
        cache[i+1] = Math.exp(tmp1);
        cache[i+2] = Math.exp(tmp2);
        cache[i+3] = Math.exp(tmp3);

        denum += (cache[i] + cache[i+1] + cache[i+2]+ cache[i+3]);
      }

      for (int i=length; i < k;++i) {
        double tmp = m != 0 ? m * Math.log(theta[i]) : 0;
        tmp += (n-m) != 0 ? (n-m) * Math.log(1-theta[i])  : 0;
        tmp += Math.log(q[i]);
        cache[i] = Math.exp(tmp);
        denum += cache[i];
      }

      for (int i=0; i < k;++i) {
        dummy.set(i,j, cache[i] != 0 && denum != 0 ? cache[i] / denum : 0);
      }

//      //test
//      {
//        double totalWeight = 0;
//        for (int i=0; i < k;++i) {
//          totalWeight += dummy.get(i,j);
//        }
//        if (Math.abs(totalWeight- 1.0) > 1e-6) {
//          System.err.println("Error: probs should sum to one");
//        }
//      }
    }
  }

  double likelihood() {
    double ll = 0;
    for (int j=0; j < sums.length;++j) {
      final double n = total[j];
      final double m = sums[j];
      double p = 0;

      final int length = (k / 4) * 4;
      for (int i=0; i < length;i+=4) {
        double tmp = m != 0 ? m * Math.log(theta[i]) : 0;
        tmp += (n-m) != 0 ? (n-m) * Math.log(1-theta[i])  : 0;
        tmp += Math.log(q[i]);
        p +=  Math.exp(tmp);
      }

      for (int i=length; i < k;++i) {
        final double t0 = theta[i];
        final double t1 = theta[i+1];
        final double t2 = theta[i+2];
        final double t3 = theta[i+3];

        final double llt0;
        final double llt1;
        final double llt2;
        final double llt3;
        if (m !=0) {
          llt0 = Math.log(t0);
          llt1 = Math.log(t1);
          llt2 = Math.log(t2);
          llt3 = Math.log(t3);
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
        if (m!=n) {
          rlt0 = Math.log(1-t0);
          rlt1 = Math.log(1-t1);
          rlt2 = Math.log(1-t2);
          rlt3 = Math.log(1-t3);
        } else {
          rlt0 = 0;
          rlt1 = 0;
          rlt2 = 0;
          rlt3 = 0;
        }

        final double  lq0 = Math.log(q[i]);
        final double  lq1 = Math.log(q[i+1]);
        final double  lq2 = Math.log(q[i+2]);
        final double  lq3 = Math.log(q[i+3]);

        final double tmp0 = m * llt0 + (n-m) * rlt0 + lq0;
        final double tmp1 = m * llt1 + (n-m) * rlt1 + lq1;
        final double tmp2 = m * llt2 + (n-m) * rlt2 + lq2;
        final double tmp3 = m * llt3 + (n-m) * rlt3 + lq3;

        final double p0 = Math.exp(tmp0);
        final double p1 = Math.exp(tmp1);
        final double p2 = Math.exp(tmp2);
        final double p3 = Math.exp(tmp3);
        final double p02 = p0 + p2;
        final double p13 = p1 + p3;
        p += p02 + p13;
      }
      ll += Math.log(p);
    }
    return ll;
  }

  boolean maximization() {
    boolean degenerate = false;
    for (int i = 0; i < k; ++i) {
      double M = 0;
      double N = 0;
      double p = 0;
      int length = (sums.length / 4) * 4;
      for (int j=0; j < length;j+=4) {
        final double prob0 = dummy.get(i,j);
        final double prob1 = dummy.get(i,j+1);
        final double prob2 = dummy.get(i,j+2);
        final double prob3 = dummy.get(i,j+3);
        final double s0= sums[j];
        final double s1= sums[j+1];
        final double s2= sums[j+2];
        final double s3= sums[j+3];
        final double t0= total[j];
        final double t1= total[j+1];
        final double t2= total[j+2];
        final double t3= total[j+3];
        final double m0 = prob0 * s0 + prob2*s2;
        final double n0 = prob0 * t0 + prob2*t2;
        final double p0 = prob0 + prob2;
        final double m1 = prob1 * s1 + prob3*s3;
        final double n1 = prob1 * t1 + prob3*t3;
        final double p1 = prob1 + prob3;
        M +=  m0 + m1;
        N += n0 + n1;
        p += p0 + p1;
      }

      for (int j= length; j  < sums.length;++j) {
        final double prob = dummy.get(i,j);
        M += prob * sums[j];
        N += prob * total[j];
        p += prob;
      }

      theta[i] = M / N;
      q[i] = p / sums.length;
    }

//    //test
//    {
//      double totalWeight = 0;
//      for (int i=0; i < q.length;++i) {
//        totalWeight += q[i];
//      }
//      if (Math.abs(totalWeight- 1.0) > 1e-6) {
//        System.err.println("Error: probs should sum to one");
//      }
//    }
    return degenerate;
  }


  Pair<double[],double[]> fit() {
    FastRandom rand = new FastRandom(42);
    {
      double totalWeight = 0;
      for (int i=0; i < q.length;++i) {
        q[i] = rand.nextDouble();
        totalWeight += q[i];
      }
      for (int i=0; i < q.length;++i) {
        q[i] /= totalWeight;
      }
    }
    {
      for (int i=0; i < theta.length;++i) {
        theta[i] = rand.nextDouble();
      }
    }
    Arrays.sort(theta);

    double[] prev = theta.clone();
//    double currentLL = Double.NEGATIVE_INFINITY;
    for (int i=0; i < 20; ++i) {
      expectation();
      if (maximization())
        break;
//      if (i % 2 == 0) {
//        double dist = l2(prev,theta);
//        if (dist < 1e-9) {
//          break;
//        }
//        prev = theta.clone();
        //optimization test
        //em should always increase likelihood
//        {
//          double ll = likelihood();
//          if (ll + 1e-9f < currentLL) {
//            System.out.println("error, em should always maximize likelihood");
//          }
//          currentLL = ll;
//        }
//      }
    }

    return new Pair<>(theta.clone(),q.clone());
  }

  private double l2(double[] a, double[] b) {
    double sum = 0;
    for (int i=0; i < a.length;++i) {
      sum += (a[i] - b[i])*(a[i] - b[i]);
    }
    return sum;
  }

}