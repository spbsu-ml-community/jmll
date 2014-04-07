package com.spbsu.ml.methods;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.SparseVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.models.ProbabilisticGraphicalModel;
import gnu.trove.map.hash.TIntDoubleHashMap;

import java.util.WeakHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 27.01.14
 * Time: 13:29
 */
public class PGMEM extends WeakListenerHolderImpl<ProbabilisticGraphicalModel> implements Optimization<LLLogit> {
  public static abstract class Policy implements Filter<ProbabilisticGraphicalModel.Route> {
    protected static final ThreadLocal<SparseVec<IntBasis>> weights = new ThreadLocal<SparseVec<IntBasis>>(){
      @Override
      protected SparseVec<IntBasis> initialValue() {
        return new SparseVec<IntBasis>(new IntBasis(Integer.MAX_VALUE));
      }
    };
    private SparseVec<IntBasis> weightsLocal;

    public SparseVec weights() {
      if (weightsLocal == null)
        weightsLocal = weights.get();
      return weightsLocal;
    }

    public Policy clear() {
      VecTools.scale(weights(), 0.);
      return this;
    }
  }

  public static final Computable<ProbabilisticGraphicalModel, Policy> MOST_PROBABLE_PATH = new Computable<ProbabilisticGraphicalModel, Policy>() {
    @Override
    public Policy compute(ProbabilisticGraphicalModel argument) {
      return new Policy() {
        public boolean accept(ProbabilisticGraphicalModel.Route route) {
          weights().set(route.index(), 1);
          return true;
        }
      }.clear();
    }
  };

  public static final Computable<ProbabilisticGraphicalModel, Policy> LAPLACE_PRIOR_PATH = new Computable<ProbabilisticGraphicalModel, Policy>() {
    @Override
    public Policy compute(ProbabilisticGraphicalModel argument) {
      return new Policy() {
        @Override
        public boolean accept(ProbabilisticGraphicalModel.Route route) {
          weights().add(route.index(), route.probab * prior(route.length()));
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
        for (ProbabilisticGraphicalModel.Route r : argument.knownRoots()){
          if (r.length() < freqs.dim())
            freqs.adjust(r.length(), r.probab);
        }
        synchronized (cache) {
          cache.put(argument, freqs);
        }
      }
      final double unknownWeight = 1 - VecTools.norm1(freqs);
      final int knownRootsCount = argument.knownRoots().length;

      final Vec finalFreqs = freqs;
      return new Policy() {
        @Override
        public boolean accept(ProbabilisticGraphicalModel.Route route) {
          final double prior = finalFreqs.get(route.length());
          weights().add(route.index(), route.probab * (prior > 0 ? prior : 2 * unknownWeight / knownRootsCount));
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

  ThreadPoolExecutor executor = ThreadTools.createBGExecutor(PGMEM.class.getName(), -1);
  @Override
  public ProbabilisticGraphicalModel fit(DataSet learn, LLLogit ll) {
    ProbabilisticGraphicalModel currentPGM = new ProbabilisticGraphicalModel(topology);
    final int[][] cpds = new int[learn.power()][];
    final Mx data = learn.data();
    for (int j = 0; j < data.rows(); j++) {
      cpds[j] = currentPGM.extractControlPoints(data.row(j));
    }

    for (int t = 0; t < iterations; t++) {
      final ProbabilisticGraphicalModel.Route[] eroutes = new ProbabilisticGraphicalModel.Route[learn.power()];
      { // E-step
        final CountDownLatch latch = new CountDownLatch(cpds.length);
        for (int j = 0; j < cpds.length; j++) {
          final ProbabilisticGraphicalModel finalCurrentPGM = currentPGM;
          final int finalJ = j;
          executor.execute(new Runnable() {
            ThreadLocal<Policy> policy = new ThreadLocal<Policy>(){
              @Override
              protected Policy initialValue() {
                return PGMEM.this.policy.compute(finalCurrentPGM);
              }
            };
            @Override
            public void run() {
              final Policy policy = this.policy.get();
              finalCurrentPGM.visit(policy, cpds[finalJ]);
              final Vec weights = policy.weights();
              final int randRoute = rng.nextSimple(weights);
              if (randRoute >= 0)
                eroutes[finalJ] = finalCurrentPGM.knownRoots()[randRoute];
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
        for (ProbabilisticGraphicalModel.Route eroute : eroutes) {
          if (eroute == null)
            continue;
          byte prev = eroute.nodes[0];
          for (int i = 1; i < eroute.nodes.length; i++) {
            next.adjust(prev, prev = eroute.nodes[i], 1.);
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
        currentPGM = new ProbabilisticGraphicalModel(next);
        invoke(currentPGM);
      }
    }
    return currentPGM;
  }
}
