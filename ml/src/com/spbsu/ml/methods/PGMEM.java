package com.spbsu.ml.methods;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.SparseVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.models.ProbabilisticGraphicalModel;

/**
 * User: solar
 * Date: 27.01.14
 * Time: 13:29
 */
public class PGMEM extends WeakListenerHolderImpl<ProbabilisticGraphicalModel> implements Optimization<LLLogit> {
  public static abstract class Policy implements Filter<ProbabilisticGraphicalModel.Route> {
    protected final SparseVec<IntBasis> weights = new SparseVec<IntBasis>(new IntBasis(Integer.MAX_VALUE));
    public Vec weights() {
      return weights;
    }

    public void clear() {
      VecTools.scale(weights, 0);
    }
  }

  public static final Policy MOST_PROBABLE_PATH = new Policy() {
    public boolean accept(ProbabilisticGraphicalModel.Route route) {
      weights.set(route.index(), 1);
      return true;
    }
  };

  public static final Policy LAPLACE_PRIOR_PATH = new Policy() {
    @Override
    public boolean accept(ProbabilisticGraphicalModel.Route route) {
      weights.add(route.index(), route.probab * prior(route.length()));
      return false;
    }
    private double prior(int length) {
      return Math.exp(-length * 0.2);
    }
  };

  private final Policy policy;
  private final Mx topology;
  private final int iterations;
  private double step;
  private final FastRandom rng;

  public PGMEM(Mx topology, double smoothing, int iterations) {
    this(topology, smoothing, iterations, new FastRandom(), MOST_PROBABLE_PATH);
  }

  public PGMEM(Mx topology, double smoothing, int iterations, FastRandom rng, Policy policy) {
    this.policy = policy;
    this.topology = topology;
    this.iterations = iterations;
    this.step = smoothing;
    this.rng = rng;
  }

  @Override
  public ProbabilisticGraphicalModel fit(DataSet learn, LLLogit ll) {
    ProbabilisticGraphicalModel currentPGM = new ProbabilisticGraphicalModel(topology);
    int[][] cpds = new int[learn.power()][];
    final Mx data = learn.data();
    for (int j = 0; j < data.rows(); j++) {
      cpds[j] = currentPGM.extractControlPoints(data.row(j));
    }
    for (int t = 0; t < iterations; t++) {
      final ProbabilisticGraphicalModel.Route[] eroutes = new ProbabilisticGraphicalModel.Route[learn.power()];
      { // E-step
        for (int j = 0; j < cpds.length; j++) {
          policy.clear();
          currentPGM.visit(policy, cpds[j]);
          if (VecTools.norm(policy.weights()) > 0)
            eroutes[j] = currentPGM.knownRoots()[rng.nextSimple(policy.weights())];
        }
      }
      final Mx next = new VecBasedMx(topology.columns(), new ArrayVec(topology.dim()));
      VecTools.fill(next, 1.); // adjusting parameters of Dir(next[i]) by one
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
