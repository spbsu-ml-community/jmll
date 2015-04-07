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
import com.spbsu.bernulli.naiveMixture.BernoulliMixtureEM;
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
  FastRandom rand = new FastRandom(22);

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
    final int k = 2;
    int tries = 100;

    final int from = 25600;
    final int to = 100000;
    final int step = 1000;

    for (int n = 320; n < 10000; n += 1000)
      for (int N = from; N < to; N += step) {
        double sumAvgMixture = 0;
        double sumAvgNaive = 0;
        for (int tr = 1; tr <= tries; ++tr) {
          BetaBinomialMixture mix = new BetaBinomialMixture(2, n, rand);
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
          System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tr + "\t" + sumAvgNaive / tr);
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
    final int k = 2;
    int tries = 100;

    final int from = 5000;
    final int to = 100001;
    final int step = 1000;

    for (int n = 30; n < 10001; n *= 10)
      for (int N = from; N < to; N *= 10) {
        double sumAvgMixture = 0;
        double sumAvgNaive = 0;
        for (int tr = 1; tr <= tries; ++tr) {
          BetaBinomialMixture mix = new BetaBinomialMixture(2, n, rand);
          final MixtureObservations observations = mix.sample(N);
          final int finaln = n;
          StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {
            @Override
            public Learner<RegularizedBetaBinomialMixtureEM> create() {
              return new Learner<RegularizedBetaBinomialMixtureEM>() {
                @Override
                public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
                  RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(k, observations.sums, finaln,200, rand);
                  FittedModel<BetaBinomialMixture> model = em.fit();
                  return new FittedModel<>(model.likelihood, em);
                }
              };
            }
          });
          RegularizedBetaBinomialMixtureEM em = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);

          double[] means = em.estimate(false);
          sumAvgMixture += observations.quality(means);
          sumAvgNaive += observations.naiveQuality() ;
          System.out.println("Real model " + mix.toString());
          System.out.println("Fitted model " + em.model().toString());
          System.out.println(tr + "\t" + n + "\t" + sumAvgMixture / tr + "\t" + sumAvgNaive / tr);

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
    final int k = 3;
    final int count = 10;
    final int n = 100;
    final int runIters = 100000000;

    NaiveMixture mixture = new NaiveMixture(k, count, rand);
    final MixtureObservations<NaiveMixture> observations = mixture.sample(n);
    final int tries = 8;
    final double[] scores = new double[tries];
    final double naiveScore = observations.naiveQuality();
    final BernoulliPrior prior = new UniformPrior(n * count + 1);
//    final BernoulliPrior prior = new LLPrior();
    System.out.println("Naive score is " + naiveScore);


//    {
//      MCMCBernoulliEstimation estimation = new MCMCBernoulliEstimation(k, observations.n, observations.sums, prior, rand);
//      int currentIters = 1;
//      for (int i=0; i < 1000;++i) {
//        estimation.run(2 * currentIters);
//        System.out.println("Score for  " + i + " is " + observations.quality(estimation.estimation()));
//        estimation.clear();
//        currentIters *= 2;
//      }
//    }
    final CountDownLatch latch = new CountDownLatch(tries);
    for (int i = 0; i < tries; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          int chainsCount = 8;
          double[] means = new double[observations.sums.length];
          for (int ii = 0; ii < chainsCount; ++ii) {
            MCMCBernoulliEstimation estimation = new MCMCBernoulliEstimation(k, observations.n, observations.sums, prior, rand);
            estimation.run(runIters);
            double[] tmp = estimation.estimation();
            System.out.println("Score for ind " + ind + " and chain " + ii + " is " + observations.quality(tmp));
            for (int j = 0; j < tmp.length; ++j)
              means[j] += tmp[j];
          }

          for (int iii = 0; iii < means.length; ++iii) {
            means[iii] /= chainsCount;
          }
          scores[ind] = observations.quality(means);
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


  public void testMCMCEstimation() {
    final int k = 5;
    final int count = 100;
    final int n = 100000;
    final int runIters = 10000000;

    NaiveMixture mixture = new NaiveMixture(k, count, rand);
    final MixtureObservations<NaiveMixture> observations = mixture.sample(n);
    final double naiveScore = observations.naiveQuality();
    final BernoulliPrior prior = new UniformPrior(n * count + 1);
//    final BernoulliPrior prior = new LLPrior();
    System.out.println("Naive score is " + naiveScore);
    MCMCBernoulliEstimation estimation = new MCMCBernoulliEstimation(k, observations.n, observations.sums, prior, rand);
    for (int i = 0; i < runIters; ++i) {
      estimation.run(10);
      double[] tmp = estimation.estimation();
      for (double p : tmp) {
        assertTrue("Probabilty shoud be in [0,1]", p >= 0 && p <= 1);
      }
    }

  }

  <Mixture> double[] fitMCMCMixture(final int k, final MixtureObservations<Mixture> experiment,final int chainsCount, final int iters) {
    final double[][] means = new double[chainsCount][experiment.sums.length];
    final BernoulliPrior prior = new UniformPrior(experiment.n * experiment.sums.length+1);
    final CountDownLatch latch = new CountDownLatch(chainsCount);
    for (int i = 0; i < chainsCount; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          MCMCBernoulliEstimation estimation = new MCMCBernoulliEstimation(k, experiment.n, experiment.sums, prior, rand);
          estimation.run(iters);
          means[ind] = estimation.estimation();
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (Exception e) {
      //
    }

    for (int i=1; i < means.length-1;++i) {
      for (int j=0; j < means[0].length;++j)
        means[0][j] += means[i][j];
    }
    final int last = means.length - 1;
    for (int j=0; j  < means[0].length;++j) {
      means[0][j] = (means[0][j] + means[last][j]) / means.length;
    }
     return means[0];
  }

  <Mixture> double[] fitBetaMixture(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {

      @Override
      public Learner<RegularizedBetaBinomialMixtureEM> create() {
        return new Learner<RegularizedBetaBinomialMixtureEM>() {
          @Override
          public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
            RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(k, experiment.sums,  experiment.n,500, rand);
            FittedModel<BetaBinomialMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em);
          }
        };
      }
    });
    RegularizedBetaBinomialMixtureEM emb = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);
    return emb.estimate(false);
  }

  <Mixture> double[] fitNaiveMixture(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<BernoulliMixtureEM> search = new StochasticSearch<>(new Factory<Learner<BernoulliMixtureEM>>() {
      @Override
      public Learner<BernoulliMixtureEM> create() {
        return new Learner<BernoulliMixtureEM>() {
          @Override
          public FittedModel<BernoulliMixtureEM> fit() {
            BernoulliMixtureEM em = new BernoulliMixtureEM(experiment.sums,  experiment.n,k, rand);
            FittedModel<NaiveMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em);
          }
        };
      }
    });
    BernoulliMixtureEM emb = search.fit(8);
    return emb.estimate(false);
  }

  public void testMixture() {
    final int k = 50;
    final int from = 500;
    final int to = 100001;
    final int tries = 1000;
    for (int n = 20; n < 1001; n *= 2)
      for (int N = from; N < to; N *= 10) {
        double sumAvgMixture = 0;
        double sumAvgBetaMixture = 0;
        double sumAvgNaive = 0;
        double sumAvgMCMC = 0;
        final NaiveMixture mixture = new NaiveMixture(k,n,rand);
        for (int tr = 1; tr <= tries; ++tr) {
          final MixtureObservations<NaiveMixture> experiment = mixture.sample(N);
          double[] betameans = fitBetaMixture(k, experiment);
          double[] mcmcmeans = fitMCMCMixture(k, experiment, 4, 100000000);
          double[] means = fitNaiveMixture(k, experiment);
          sumAvgMixture += experiment.quality(means);
          sumAvgBetaMixture +=experiment.quality(betameans);
          sumAvgNaive += experiment.naiveQuality();
          sumAvgMCMC += experiment.quality(mcmcmeans);
          System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tr + "\t" + sumAvgBetaMixture / tr+ "\t" +sumAvgMCMC / tr + "\t"  + sumAvgNaive / tr);
        }
        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + sumAvgBetaMixture / tries + "\t" +sumAvgMCMC / tries + "\t"  + sumAvgNaive / tries);
      }
  }


  public void testNaiveMixture() {
    final int k = 2;
    final int from = 6400;
    final int to = 100001;
    final int tries = 1000;
    for (int n = 320; n < 1001; n *= 2)
      for (int N = from; N < to; N *= 10) {
        double sumAvgMixture = 0;
        double sumAvgNaive = 0;
        final NaiveMixture mixture = new NaiveMixture(k,n,rand);
        for (int tr = 1; tr <= tries; ++tr) {
          final MixtureObservations<NaiveMixture> experiment = mixture.sample(N);
          double[] means = fitNaiveMixture(k, experiment);
          sumAvgMixture += experiment.quality(means);
          sumAvgNaive += experiment.naiveQuality();
          System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tr + "\t" + "\t"  + sumAvgNaive / tr);
        }
        System.out.println(N + "\t" + n + "\t" + sumAvgMixture / tries + "\t" + "\t"  + sumAvgNaive / tries);
      }
  }
}
