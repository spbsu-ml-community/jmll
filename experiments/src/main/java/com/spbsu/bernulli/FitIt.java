package com.spbsu.bernulli;

/**
 * Created by noxoomo on 13/06/15.
 */

import com.spbsu.bernulli.MCMCBernoulliMixture.BernoulliPrior;
import com.spbsu.bernulli.MCMCBernoulliMixture.MCMCBernoulliEstimation;
import com.spbsu.bernulli.MCMCBernoulliMixture.UniformPrior;
import com.spbsu.bernulli.betaBinomialMixture.BetaBinomialMixture;
import com.spbsu.bernulli.betaBinomialMixture.BetaBinomialMixtureEM;
import com.spbsu.bernulli.betaBinomialMixture.RegularizedBetaBinomialMixtureEM;
import com.spbsu.bernulli.dirichletMixture.DirichletEstimation;
import com.spbsu.bernulli.naiveMixture.BernoulliMixtureEM;
import com.spbsu.bernulli.naiveMixture.NaiveMixture;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 11/04/15.
 */
public class FitIt {
  static FastRandom rand = new FastRandom(0);
  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("FitITExecutor", -1);

  public static <Mixture> double[] fitBetaMixture(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<BetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<BetaBinomialMixtureEM>>() {
      @Override
      public Learner<BetaBinomialMixtureEM> create() {
        return new Learner<BetaBinomialMixtureEM>() {
          @Override
          public FittedModel<BetaBinomialMixtureEM> fit() {
            BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k, experiment.sums, experiment.n, rand);
            FittedModel<BetaBinomialMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em, k * 3);
          }
        };
      }
    });
    BetaBinomialMixtureEM emb = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);
    return emb.estimate(false);
  }


  public static <Mixture> BetaBinomialMixtureEM fitBetaMixture2(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<BetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<BetaBinomialMixtureEM>>() {
      @Override
      public Learner<BetaBinomialMixtureEM> create() {
        return new Learner<BetaBinomialMixtureEM>() {
          @Override
          public FittedModel<BetaBinomialMixtureEM> fit() {
            BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k, experiment.sums, experiment.n, rand);
            FittedModel<BetaBinomialMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em, k * 3);
          }
        };
      }
    });
    BetaBinomialMixtureEM emb = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);
    return emb;
  }


  public static <Mixture> RegularizedBetaBinomialMixtureEM fitRegBetaMixture2(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {
      @Override
      public Learner<RegularizedBetaBinomialMixtureEM> create() {
        return new Learner<RegularizedBetaBinomialMixtureEM>() {
          @Override
          public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
            RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(k, experiment.sums, experiment.n, 500, rand);
            FittedModel<BetaBinomialMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em, k * 3);
          }
        };
      }
    });
    RegularizedBetaBinomialMixtureEM emb = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);
    return emb;
  }


  public static <Mixture> double[] fitRegBetaMixture(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<RegularizedBetaBinomialMixtureEM> search = new StochasticSearch<>(new Factory<Learner<RegularizedBetaBinomialMixtureEM>>() {
      @Override
      public Learner<RegularizedBetaBinomialMixtureEM> create() {
        return new Learner<RegularizedBetaBinomialMixtureEM>() {
          @Override
          public FittedModel<RegularizedBetaBinomialMixtureEM> fit() {
            RegularizedBetaBinomialMixtureEM em = new RegularizedBetaBinomialMixtureEM(k, experiment.sums, experiment.n, 500, rand);
            FittedModel<BetaBinomialMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em, k * 3);
          }
        };
      }
    });
    RegularizedBetaBinomialMixtureEM emb = search.fit(8);
//          BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(k,observations.sums,n,rand);
    return emb.estimate(false);
  }

  public static <Mixture> double[] fitNaiveMixture(final int k, final MixtureObservations<Mixture> experiment) {
    StochasticSearch<BernoulliMixtureEM> search = new StochasticSearch<>(new Factory<Learner<BernoulliMixtureEM>>() {
      @Override
      public Learner<BernoulliMixtureEM> create() {
        return new Learner<BernoulliMixtureEM>() {
          @Override
          public FittedModel<BernoulliMixtureEM> fit() {
            BernoulliMixtureEM em = new BernoulliMixtureEM(experiment.sums, experiment.n, k, rand);
            FittedModel<NaiveMixture> model = em.fit();
            return new FittedModel<>(model.likelihood, em, 2 * k);
          }
        };
      }
    });
    BernoulliMixtureEM emb = search.fit(8);
    return emb.estimate(false);
  }


  public static final int gibbsIters = 2000;
  public static final int gibbsWindow = 2;

  public static <Mixture> double[] fitDirichlet(final MixtureObservations<Mixture> experiment) {
    DirichletEstimation estimation = new DirichletEstimation(1, experiment.n, experiment.sums, rand);
    estimation.burnIn();
    estimation.run(gibbsIters, gibbsWindow);
    return estimation.estimate();
  }

  public static <Mixture> double[] fitAicEm(final MixtureObservations<Mixture> experiment) {
    AicEM<FittedModel<BernoulliMixtureEM>> estimation = new AicEM<>(i -> {
      BernoulliMixtureEM em = new BernoulliMixtureEM(experiment.sums, experiment.n, i, rand);
      FittedModel<NaiveMixture> model = em.fit();
      return new FittedModel<>(model.likelihood, em, 2 * i);
    });
    return estimation.fit().model.estimate(false);
  }


  public static <Mixture> double[] fitAicBetaEm(final MixtureObservations<Mixture> experiment) {
    AicEM<FittedModel<BetaBinomialMixtureEM>> estimation = new AicEM<>(i -> {
      BetaBinomialMixtureEM em = new BetaBinomialMixtureEM(i, experiment.sums, experiment.n, rand);
      FittedModel<BetaBinomialMixture> model = em.fit();
      return new FittedModel<>(model.likelihood, em, i * 3);
    });
    return estimation.fit().model.estimate(false);
  }


  public static <Mixture> double[] fitMCMCMixture(final int k, final MixtureObservations<Mixture> experiment, final int chainsCount, final int iters) {
    final double[][] means = new double[chainsCount][experiment.sums.length];
    final BernoulliPrior prior = new UniformPrior(experiment.n * experiment.sums.length + 1);
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

    for (int i = 1; i < means.length - 1; ++i) {
      for (int j = 0; j < means[0].length; ++j)
        means[0][j] += means[i][j];
    }
    final int last = means.length - 1;
    for (int j = 0; j < means[0].length; ++j) {
      means[0][j] = (means[0][j] + means[last][j]) / means.length;
    }
    return means[0];
  }
}

