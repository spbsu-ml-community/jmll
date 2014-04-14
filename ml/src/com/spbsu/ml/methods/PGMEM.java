package com.spbsu.ml.methods;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.SparseVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.models.pgm.ProbabilisticGraphicalModel;
import com.spbsu.ml.models.pgm.Route;
import com.spbsu.ml.models.pgm.SimplePGM;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.WeakHashMap;
import java.util.concurrent.*;

/**
 * User: solar
 * Date: 27.01.14
 * Time: 13:29
 */
public class PGMEM extends WeakListenerHolderImpl<SimplePGM> implements Optimization<LLLogit> {
  public static abstract class Policy implements Filter<Route> {
    private final Vec weights;
    private final List<Route> routes;

    protected Policy(ProbabilisticGraphicalModel pgm) {
      weights = new ArrayVec(pgm.knownRoutesCount());
      routes = new ArrayList<Route>(pgm.knownRoutesCount());
    }
    protected void addOption(Route r, double w) {
      weights.set(routes.size(), w);
      routes.add(r);
    }

    public Route next(FastRandom rng) {
      return routes.isEmpty() ? null : routes.get(rng.nextSimple(weights));
    }

    public Policy clear() {
      VecTools.scale(weights, 0.);
      routes.clear();
      return this;
    }
  }

  public static final Computable<ProbabilisticGraphicalModel, Policy> MOST_PROBABLE_PATH = new Computable<ProbabilisticGraphicalModel, Policy>() {
    @Override
    public Policy compute(ProbabilisticGraphicalModel argument) {
      return new Policy(argument) {
        public boolean accept(Route route) {
          addOption(route, 1.);
          return true;
        }
      }.clear();
    }
  };

  public static final Computable<ProbabilisticGraphicalModel, Policy> LAPLACE_PRIOR_PATH = new Computable<ProbabilisticGraphicalModel, Policy>() {
    @Override
    public Policy compute(ProbabilisticGraphicalModel argument) {
      return new Policy(argument) {
        @Override
        public boolean accept(Route route) {
          addOption(route, route.p() * prior(route.length()));
          return false;
        }
        private double prior(int length) {
          return Math.exp(-length);
        }
      }.clear();
    }
  };

  public static final Computable<ProbabilisticGraphicalModel, Policy> FREQ_DENSITY_PRIOR_PATH = new Computable<ProbabilisticGraphicalModel, Policy>() {
    final WeakHashMap<ProbabilisticGraphicalModel, Vec> cache = new WeakHashMap<ProbabilisticGraphicalModel, Vec>();
    @Override
    public Policy compute(ProbabilisticGraphicalModel argument) {
      Vec freqs = cache.get(argument);
      if (freqs == null) {
        freqs = new ArrayVec(10000);
        for (int i = 0; i < argument.knownRoutesCount(); i++) {
          Route r = argument.knownRoute(i);
          if (r.length() < freqs.dim())
            freqs.adjust(r.length(), r.p());
        }
        synchronized (cache) {
          cache.put(argument, freqs);
        }
      }
      final double unknownWeight = 1 - VecTools.norm1(freqs);
      final int knownRootsCount = argument.knownRoutesCount();

      final Vec finalFreqs = freqs;
      return new Policy(argument) {
        @Override
        public boolean accept(Route route) {
          final double prior = finalFreqs.get(route.length());
          addOption(route, route.p() * (prior > 0 ? prior : 2 * unknownWeight / knownRootsCount));
          return false;
        }
      }.clear();
    }
  };

  private final Computable<ProbabilisticGraphicalModel, Policy> policy;
  private final Mx topology;
  private final int iterations;
  private double step;
  private final FastRandom rng;

  public PGMEM(Mx topology, double smoothing, int iterations) {
    this(topology, smoothing, iterations, new FastRandom(), MOST_PROBABLE_PATH);
  }

  public PGMEM(Mx topology, double smoothing, int iterations, FastRandom rng, Computable<ProbabilisticGraphicalModel, Policy> policy) {
    this.policy = policy;
    this.topology = topology;
    this.iterations = iterations;
    this.step = smoothing;
    this.rng = rng;
  }

  @Override
  public SimplePGM fit(DataSet learn, LLLogit ll) {
    final ThreadGroup tg = new ThreadGroup(PGMEM.class.getName());
    final Thread[] threads = new Thread[ThreadTools.COMPUTE_UNITS];
    final ArrayBlockingQueue<Action<Policy>> queue = new ArrayBlockingQueue<Action<Policy>>(learn.power());
    final Policy[] policies = new Policy[threads.length];
    for (int i = 0; i < threads.length; i++) {
      final int finalI = i;
      threads[i] = new Thread(tg, new Runnable() {
        @Override
        public void run() {
          try {
            while(true) {
              final Action<Policy> next = queue.take();
              policies[finalI].clear();
              next.invoke(policies[finalI]);
            }
          } catch (InterruptedException e) {
            //
          }
        }
      });
      threads[i].start();
    }

    SimplePGM currentPGM = new SimplePGM(topology);
    final int[][] cpds = new int[learn.power()][];
    final Mx data = learn.data();
    for (int j = 0; j < data.rows(); j++) {
      cpds[j] = currentPGM.extractControlPoints(data.row(j));
    }

    for (int t = 0; t < iterations; t++) {
      final Route[] eroutes = new Route[learn.power()];
      { // updating policies
        for (int i = 0; i < policies.length; i++) {
          policies[i] = policy.compute(currentPGM);
        }
      }
      { // E-step
        final CountDownLatch latch = new CountDownLatch(cpds.length);

        for (int j = 0; j < cpds.length; j++) {
          final SimplePGM finalCurrentPGM = currentPGM;
          final int finalJ = j;
          queue.add(new Action<Policy>() {
            @Override
            public void invoke(Policy policy) {
              finalCurrentPGM.visit(policy, cpds[finalJ]);
              eroutes[finalJ] = policy.next(rng);
              latch.countDown();
            }
          });
        }
        try {
          latch.await();
        } catch (InterruptedException e) {
          // skip
        }
      }

      final Mx next = new VecBasedMx(topology.columns(), new ArrayVec(topology.dim()));
      { // adjusting parameters of Dir(next[i]) by one only if this way is possible
        final MxIterator it = topology.nonZeroes();
        while (it.advance()) {
          if (it.value() > MathTools.EPSILON)
            next.adjust(it.index(), 1.);
        }
      }
      { // M-step
        for (Route eroute : eroutes) {
          if (eroute == null)
            continue;
          int prev = eroute.dst(0);
          for (int i = 1; i < eroute.length(); i++) {
            next.adjust(prev, prev = eroute.dst(i), 1.);
          }
        }
        for (int i = 0; i < next.rows(); i++) {
          VecTools.normalizeL1(next.row(i)); // assuming weights of nodes are distributed by Dir(next[i]), then optimal parameters will be proportional to pass count
        }
      }
      { // Update PGM
        VecTools.scale(next, step/(1. - step));
        VecTools.append(next, currentPGM.topology);
        VecTools.scale(next, (1. - step));
        currentPGM = new SimplePGM(next);
        invoke(currentPGM);
      }
    }
    tg.interrupt();
    return currentPGM;
  }
}
