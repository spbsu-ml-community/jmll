package com.spbsu.bernulli;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;

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
    final int tries = 10;
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
        prob = em.dummy.get(i,j) * em.theta[i];
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
  double[] cache;

  double[] q;
  double[] theta;

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
      for (int i=0; i < k;++i) {
        double tmp = m != 0 ? m * Math.log(theta[i]) : 0;
        tmp += (n-m) != 0 ? (n-m) * Math.log(1-theta[i])  : 0;
        tmp += Math.log(q[i]);
        cache[i] = Math.exp(tmp);
        denum += cache[i];
      }

      for (int i=0; i < k;++i) {
        dummy.set(i,j, cache[i] / denum);
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
      for (int i=0; i < k;++i) {
        double tmp = m != 0 ? m * Math.log(theta[i]) : 0;
        tmp += (n-m) != 0 ? (n-m) * Math.log(1-theta[i])  : 0;
        tmp += Math.log(q[i]);
        p +=  Math.exp(tmp);
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
      for (int j=0; j < sums.length;++j) {
        final double prob = dummy.get(i,j);
        M += prob * sums[j];
        N += prob * total[j];
        p += prob;
      }

      theta[i] = M / N;
      q[i] = p / sums.length;
      if (theta[i] == 0 || theta[i] == 1) {
        degenerate = true;
      }
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
    FastRandom rand = new FastRandom();
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

    double[] prev = theta.clone();
//    double currentLL = Double.NEGATIVE_INFINITY;
    for (int i=0; i < 200; ++i) {
      expectation();
      if (maximization())
        break;
      if (i % 5 == 0) {
        double dist = l2(prev,theta);
        if (dist < 1e-9) {
          break;
        }
        prev = theta.clone();
        //optimization test
        //em should always increase likelihood
//        {
//          double ll = likelihood();
//          if (ll + 1e-9f < currentLL) {
//            System.out.println("error, em should always maximize likelihood");
//          }
//          currentLL = ll;
//        }
      }
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