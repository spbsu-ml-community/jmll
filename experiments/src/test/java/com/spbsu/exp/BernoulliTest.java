package com.spbsu.exp;

import com.spbsu.bernulli.FittedModel;
import com.spbsu.bernulli.Learner;
import com.spbsu.bernulli.MCMCBernoulliMixture.BernoulliPrior;
import com.spbsu.bernulli.MCMCBernoulliMixture.MCMCBernoulliEstimation;
import com.spbsu.bernulli.MCMCBernoulliMixture.UniformPrior;
import com.spbsu.bernulli.MixtureObservations;
import com.spbsu.bernulli.StochasticSearch;
import com.spbsu.bernulli.betaBinomialMixture.BetaBinomialMixture;
import com.spbsu.bernulli.betaBinomialMixture.BetaBinomialMixtureEM;
import com.spbsu.bernulli.betaBinomialMixture.RegularizedBetaBinomialMixtureEM;
import com.spbsu.bernulli.naiveMixture.NaiveMixture;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import junit.framework.TestCase;
import org.apache.commons.math3.special.Gamma;

import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.bernulli.betaBinomialMixture.BetaBinomialMixtureEM.Type;
import static com.spbsu.commons.math.MathTools.sqr;
import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * bernoulli models experiments
 * User: noxoomo
 * Date: 09.03.2015
 */
public class BernoulliTest extends TestCase {
  FastRandom rand = new FastRandom(42);

  public void testBetaBinomialMixture() {
    final int k = 3;
    final int n = 200;
    final int count = 5000;
    int tries = 200;

    for (int i = 0; i < tries; ++i) {
      BetaBinomialMixture mixture = new BetaBinomialMixture(k, count, rand);
      MixtureObservations observation = mixture.sample(n);
      BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k, observation.sums, n, rand);
      FittedModel<BetaBinomialMixture> fittedModel = em.fit(true);
      System.out.println("Real model " + mixture);
      System.out.println("Fitted model " + fittedModel.model);
    }
  }

  public void testBetaBinomialRegularizedMixture() {
    final int k = 3;
    final int n = 200;
    final int count = 5000;
    int tries = 200;
    int fakeObservations = 1000;

    for (int i = 0; i < tries; ++i) {
      BetaBinomialMixture mixture = new BetaBinomialMixture(k, count, rand);
      MixtureObservations observation = mixture.sample(n);
      RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(k, observation.sums, n, fakeObservations, rand);
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
          BetaBinomialMixture mix = new BetaBinomialMixture(5, n, rand);
          final MixtureObservations observations = mix.sample(N);
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
          sumAvgMixture += observations.quality(means) / observations.thetas.length;
          sumAvgNaive += observations.naiveQuality() / observations.thetas.length;
        }
        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + sumAvgNaive / tries);
      }
  }

  public void testSteinBernoulli() {
    Random rng = new FastRandom();
    for (int n = 50; n < 1001; n *= 2) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumAvgBetaBin = 0;
      for (int k = 0; k < 1000; k++) {
        Vec sum = new ArrayVec(1000);
        Vec m = new ArrayVec(1000);
        for (int i = 0; i < m.dim(); ++i)
          m.set(i, rng.nextDouble());
        double sigma = 1;

        for (int i = 0; i < n; i++) {
          for (int t = 0; t < sum.dim(); t++) {
            sum.adjust(t, rng.nextDouble() > m.get(t) ? 0 : 1);
          }
        }
        Vec naive = copy(sum);
        scale(naive, 1. / n);
        Vec js = copy(sum);
        scale(js, (1 - (js.dim() - 2) * sigma * sigma / sqr(norm(sum))) / n);

        double[] betameans;
        {
          final int[] isums = new int[sum.dim()];
          for (int i = 0; i < isums.length; ++i) {
            isums[i] = (int) sum.get(i);
          }

          final int finalN = n;
          StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {

            @Override
            public Learner<RegularizedBetaBinomialMixtureEM> create() {
              return new Learner<RegularizedBetaBinomialMixtureEM>() {
                @Override
                public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
                  RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(4, isums, finalN, 100, rand);
                  FittedModel<BetaBinomialMixture> model = em.fit();
                  return new FittedModel<>(model.likelihood, em);
                }
              };
            }
          });
          RegularizedBetaBinomialMixtureEM emb = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);

          betameans = emb.estimate(false);
        }

        sumAvgNaive += distance(naive, m) / Math.sqrt(m.dim());
        sumAvgJS += distance(js, m) / Math.sqrt(m.dim());
        sumAvgBetaBin += distance(new ArrayVec(betameans), m) / Math.sqrt(m.dim());
      }
      System.out.println(n + "\t" + sumAvgNaive / 1000 + "\t" + sumAvgJS / 1000 + "\t" + sumAvgBetaBin / 1000);
    }
  }

  public void testBetaBinomialRegularizedMixtureEstimation() {
    final int k = 5;
    int tries = 100;

    final int from = 100;
    final int to = 10001;
    final int step = 1000;

    for (int n = 10; n < 10001; n *= 10)
      for (int N = from; N < to; N *= 10) {
        double sumAvgMixture = 0;
        double sumAvgNaive = 0;
        for (int tr = 0; tr < tries; ++tr) {
          BetaBinomialMixture mix = new BetaBinomialMixture(16, n, rand);
          final MixtureObservations observations = mix.sample(N);
          final int finaln = n;
          StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {
            @Override
            public Learner<RegularizedBetaBinomialMixtureEM> create() {
              return new Learner<RegularizedBetaBinomialMixtureEM>() {
                @Override
                public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
                  RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(k, observations.sums, finaln, 200, rand);
                  FittedModel<BetaBinomialMixture> model = em.fit();
                  return new FittedModel<>(model.likelihood, em);
                }
              };
            }
          });
          RegularizedBetaBinomialMixtureEM em = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);

          double[] means = em.estimate(false);
          sumAvgMixture += observations.quality(means) / observations.thetas.length;
          sumAvgNaive += observations.naiveQuality() / observations.thetas.length;
        }
        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + sumAvgNaive / tries);
      }
  }


  public void testSpecialFunctionCache() {
//    BetaFunctionsProportion prop = new  BetaFunctionsProportion(0.5,0.5,50);
    BetaBinomialMixtureEM.SpecialFunctionCache prop = new BetaBinomialMixtureEM.SpecialFunctionCache(1, 1, 10);
//    assertTrue((prop.calculate(10,50)-1.860013246596769 * 1e-13) < 1e-20);
//    prop.update(1,1);
    assertTrue(Math.abs(prop.calculate(2, 10) - 0.0020202) < 1e-7);
    assertTrue(Math.abs(prop.calculate(3, 10) - 0.000757576) < 1e-7);
    assertTrue(Math.abs(prop.calculate(4, 10) - 0.0004329) < 1e-7);
    assertTrue(Math.abs(prop.calculate(5, 10) - 0.00036075) < 1e-7);
    assertTrue(Math.abs(prop.calculate(9, 10) - 0.00909091) < 1e-7);
    assertTrue(Math.abs(prop.calculate(10, 10) - 0.0909091) < 1e-7);

    for (int i = 0; i < 10; ++i) {
      assertTrue(Math.abs(prop.digamma(Type.Alpha, i) - Gamma.digamma(1 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(Type.Beta, i) - Gamma.digamma(1 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(Type.AlphaBeta, i) - Gamma.digamma(2 + i)) < 1e-12);
    }

    prop.update(102.5, 10.11);
//    prop = new  BetaFunctionsProportion(102.5,10.11,10);
    assertTrue(Math.abs(prop.calculate(5, 10) - 6.478713223568241e-6) < 1e-9);
    assertTrue(Math.abs(prop.calculate(8, 10) - 0.00369375) < 1e-8);

    for (int i = 0; i < 10; ++i) {
      assertTrue(Math.abs(prop.digamma(Type.Alpha, i) - Gamma.digamma(102.5 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(Type.Beta, i) - Gamma.digamma(10.11 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(Type.AlphaBeta, i) - Gamma.digamma(102.5 + 10.11 + i)) < 1e-12);
    }

    for (int i = 0; i < 10; ++i) {
      assertTrue(Math.abs(prop.digamma1(Type.Alpha, i) - Gamma.trigamma(102.5 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma1(Type.Beta, i) - Gamma.trigamma(10.11 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma1(Type.AlphaBeta, i) - Gamma.trigamma(102.5 + 10.11 + i)) < 1e-12);
    }

    prop = new BetaBinomialMixtureEM.SpecialFunctionCache(12.2, 55.1, 100);

    assertTrue(Math.abs(prop.calculate(30, 100) - 3.520721627628687e-28) < 1e-32);
    assertTrue(Math.abs(prop.calculate(60, 100) - 7.007620723590574e-37) < 1e-41);
    assertTrue(Math.abs(prop.calculate(10, 100) - 1.764175281317258e-15) < 1e-19);
    assertTrue(Math.abs(prop.calculate(88, 100) - 3.97033387162681e-36) < 1e-40);


    for (int i = 0; i < 100; ++i) {
      assertTrue(Math.abs(prop.digamma(Type.Alpha, i) - Gamma.digamma(12.2 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(Type.Beta, i) - Gamma.digamma(55.1 + i)) < 1e-12);
      assertTrue(Math.abs(prop.digamma(Type.AlphaBeta, i) - Gamma.digamma(12.2 + 55.1 + i)) < 1e-12);
    }
  }

  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("bernoulli tests", -1);

  public void testMCMCConvergence() {
    final int k = 10;
    final int count = 200;
    final int n = 20000;
    final int runIters = 10000000;

    NaiveMixture mixture = new NaiveMixture(k, count, rand);
    final MixtureObservations<NaiveMixture> observations = mixture.sample(n);
    final int tries = 20;
    final double[] scores = new double[tries];
    final double naiveScore = observations.naiveQuality();
    final BernoulliPrior prior = new UniformPrior(n * count + 1);
//    final BernoulliPrior prior = new LLPrior();

//    {
//      MCMCBernoulliEstimation estimation = new MCMCBernoulliEstimation(k, observations.n, observations.sums, prior, rand);
//      int currentIters = 1;
//      for (int i=0; i < 20;++i) {
//        estimation.run(2 * currentIters);
//        System.out.println("Score for  " + i + " is " + observations.quality(estimation.estimation()));
//        currentIters *= 2;
//      }
//    }
    final CountDownLatch latch = new CountDownLatch(tries);
    System.out.println("Naive score is " + naiveScore);
    for (int i = 0; i < tries; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          MCMCBernoulliEstimation estimation = new MCMCBernoulliEstimation(k, observations.n, observations.sums, prior, rand);
          estimation.run(runIters);
          scores[ind] = observations.quality(estimation.estimation());
          System.out.println("Score for " + ind + " is " + scores[ind]);
          latch.countDown();
        }
      });

    }
    try {
      latch.await();
    } catch (Exception e) {
      //
    }

  }
//  public void testMixture() {
//    final int k = 20;
//    final int from = 100;
//    final int to = 10001;
//    final int step = 50;
//    final int tries = 1000;
//    BernoulliRand brand = new BernoulliRand();
//
//
//    double[] theta = new double[k];
//    double[] q = new double[k];
//    for (int n = 10; n < 1001; n *= 10)
//      for (int N = from; N < to; N *= 10) {
//        double sumAvgMixture = 0;
//        double sumAvgBetaMixture = 0;
//        double sumAvgNaive = 0;
//        for (int tr = 0; tr < tries; ++tr) {
//          {
//            double totalWeight = 0;
//            for (int i = 0; i < q.length; ++i) {
//              q[i] = rand.nextDouble();
//              totalWeight += q[i];
//            }
//            for (int i = 0; i < q.length; ++i) {
//              q[i] /= totalWeight;
//            }
//          }
//          {
//            for (int i = 0; i < theta.length; ++i) {
//              theta[i] = rand.nextDouble() / 4;
//            }
//          }
//          final BernoulliRand.Result experiment = brand.simulate(theta, q, N, n);
//          BernoulliMixture em = new BernoulliMixture();
//
//          final  int[] isums = new int[experiment.sums.length];
//          for (int i=0; i < isums.length;++i) {
//                  isums[i] = (int) experiment.sums[i];
//          }
//
//          StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {
//
//            @Override
//            public Learner<RegularizedBetaBinomialMixtureEM> create() {
//              return new Learner<RegularizedBetaBinomialMixtureEM>() {
//                @Override
//                public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
//                  RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(5, isums, (int) experiment.total[0],500, rand);
//                  FittedModel<BetaBinomialMixture> model = em.fit();
//                  return new FittedModel<>(model.likelihood, em);
//                }
//              };
//            }
//          });
//          RegularizedBetaBinomialMixtureEM emb = search.fit(8);
////          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);
//
//          double[] betameans = emb.estimate(false);
//
//
//          double[] means = em.estimate(experiment.sums, experiment.total, k);
//          double[] naive = experiment.sums.clone();
//          for (int i = 0; i < naive.length; ++i) {
//            naive[i] /= experiment.total[i];
//          }
//          sumAvgMixture += l2(means, experiment.means) / experiment.means.length;
//          sumAvgBetaMixture += l2(betameans, experiment.means) / experiment.means.length;
//          sumAvgNaive += l2(naive, experiment.means) / experiment.means.length;
//        }
//        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + sumAvgBetaMixture / tries + "\t" + sumAvgNaive / tries);
//      }
//  }


}
