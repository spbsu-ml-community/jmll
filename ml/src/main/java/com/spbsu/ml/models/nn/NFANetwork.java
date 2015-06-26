package com.spbsu.ml.models.nn;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.func.generic.SubVecFuncC1;
import com.spbsu.ml.func.generic.WSum;
import com.spbsu.ml.models.NeuralSpider;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 13:25
 */
public class NFANetwork<T> extends NeuralSpider<T, Seq<T>> {
  private final FastRandom rng;
  private final double dropout;
  private final int statesCount;
  private final Seq<T> alpha;
  private final int dim;
  private final int transitionMxDim;

  public NFANetwork(FastRandom rng, double dropout, int statesCount, Seq<T> alpha) {
    super();
    this.rng = rng;
    this.dropout = dropout;
    this.statesCount = statesCount;
    this.alpha = alpha;
    transitionMxDim = (statesCount - 1) * (statesCount - 1);
    dim = transitionMxDim * alpha.length();
  }

  private static class WeightsCalculator implements Computable<Vec,Mx> {
    private final int statesCount;
    private final int wStart;
    private final int wLen;
    private boolean[] dropOut;

    private WeightsCalculator(int statesCount, int wStart, int wLen) {
      this.statesCount = statesCount;
      this.wStart = wStart;
      this.wLen = wLen;
    }

    final ThreadLocalArrayVec w = new ThreadLocalArrayVec();
    public Mx computeInner(Vec betta) {
      final VecBasedMx b = new VecBasedMx(statesCount - 1, betta.sub(wStart, wLen));
      final VecBasedMx w = new VecBasedMx(statesCount, this.w.get(statesCount * statesCount));
      for (int i = 0; i < statesCount - 1; i++) {
        if (dropOut[i])
          continue;
        double sum = 1;
        for (int j = 0; j < statesCount - 1; j++) {
          if (dropOut[j])
            continue;
          sum += Math.exp(b.get(i, j));
        }
        for (int j = 0; j < statesCount; j++) {
          if (dropOut[j])
            continue;
          final double selectedExp = j < statesCount - 1 ? Math.exp(b.get(i, j)) : 1;
          w.set(j, i, selectedExp / sum);
        }
      }
      return w;
    }

    private Vec cacheArg;
    private Mx cacheVal;
    @Override
    public Mx compute(Vec betta) {
      if (!betta.isImmutable())
        return computeInner(betta);
      if (betta == cacheArg)
        return cacheVal;
      cacheArg = betta;
      return cacheVal = computeInner(betta);
    }

    public void setDropOut(boolean[] dropOut) {
      this.dropOut = dropOut;
    }
  }

  final ThreadLocal<WeightsCalculator[]> calculators = new ThreadLocal<WeightsCalculator[]>() {
    @Override
    protected WeightsCalculator[] initialValue() {
      final WeightsCalculator[] calculators = new WeightsCalculator[alpha.length()];
      for(int i = 0; i < calculators.length; i++) {
        calculators[i] = new WeightsCalculator(statesCount, i * transitionMxDim, transitionMxDim);
      }
      return calculators;
    }
  };

  @Override
  protected Topology topology(Seq<T> seq, final boolean dropout) {
    final Node[] nodes = new Node[(seq.length() + 1) * statesCount + 1];
    for (int i = 0; i < statesCount; i++) {
      final Const aConst = new Const(i == 0 ? 1 : 0);
      nodes[i] = new Node() {
        @Override
        public FuncC1 transByParameters(Vec betta) {
          return aConst;
        }

        @Override
        public FuncC1 transByParents(Vec state) {
          return aConst;
        }
      };
    }
    final boolean[] dropoutArr = new boolean[statesCount];

    if (dropout && rng.nextDouble() < this.dropout)
      dropoutArr[rng.nextInt(statesCount - 2) + 1] = true;

    final int[] lastNodeConnections = new int[seq.length()];
    for (int d = 0; d < seq.length(); d++) {
      final int elementIndex = ArrayTools.indexOf(seq.at(d), alpha);
      final int prevLayerStart = d * statesCount;
      final WeightsCalculator calcer = calculators.get()[elementIndex];
      calcer.setDropOut(dropoutArr);
      for (int i = 0; i < statesCount; i++) {
        final int nodeIndex = (d + 1) * statesCount + i;
        nodes[nodeIndex] = new MyNode(i, elementIndex * transitionMxDim, transitionMxDim, prevLayerStart, statesCount, nodes.length + statesCount, calcer);
      }
      lastNodeConnections[d] = (d + 2) * statesCount - 1;
    }
    nodes[nodes.length - 1] = new Node() {
      @Override
      public FuncC1 transByParameters(final Vec betta) {
        return new FuncC1.Stub() {
          @Override
          public Vec gradientTo(Vec x, Vec to) {
            for (int i = 0; i < lastNodeConnections.length; i++) {
              to.set(lastNodeConnections[i], 1);
            }
            return to;
          }

          @Override
          public double value(Vec x) {
            double sum = 0;
            for (int i = 0; i < lastNodeConnections.length; i++) {
              sum += x.get(lastNodeConnections[i]);
            }
            return sum;
          }

          @Override
          public int dim() {
            return nodes.length;
          }
        };
      }

      @Override
      public FuncC1 transByParents(Vec state) {
        double sum = 0;
        for (int i = 0; i < lastNodeConnections.length; i++) {
          sum += state.get(lastNodeConnections[i]);
        }
        return new Const(sum);
      }
    };

    return new Topology.Stub() {
      @Override
      public int outputCount() {
        return 1;
      }

      @Override
      public boolean isDroppedOut(int nodeIndex) {
        if (!dropout || nodeIndex == 0 || nodeIndex == nodes.length - 1)
          return false;
        final int state = nodeIndex % statesCount;
        return dropoutArr[state];
      }

      @Override
      public Node at(int i) {
        return nodes[i];
      }

      @Override
      public int length() {
        return nodes.length;
      }
    };
  }

  @Override
  public int dim() {
    return dim;
  }

  public String ppSolution(Vec x) {
    final StringBuilder builder = new StringBuilder();
    for (int i = 0; i < alpha.length(); i++) {
      builder.append(alpha.at(i)).append(":").append("\n");
      builder.append(new VecBasedMx(statesCount - 1, x.sub(i * transitionMxDim, transitionMxDim))).append("\n");
    }
    return builder.toString();
  }

  public String ppState(Vec state, Seq<T> seq) {
    final StringBuilder builder = new StringBuilder();
    for (int i = 0; i < state.dim() / statesCount; i++) {
      if (seq != null && i > 0 && i <= seq.length())
        builder.append(seq.at(i - 1));
      else
        builder.append(" ");

      for (int s = 0; s < statesCount; s++) {
        builder.append("\t").append(CharSeqTools.prettyPrint.format(state.get(i * statesCount + s)));
      }
      builder.append('\n');
    }
    builder.append(" \t").append(CharSeqTools.prettyPrint.format(state.get(state.dim() - 1))).append('\n');
    return builder.toString();
  }

  public String ppSolution(Vec x, T s) {
    final int i = SeqTools.indexOf(alpha, s);
    return String.valueOf(s) + ":\n" + new VecBasedMx(statesCount - 1, x.sub(i * transitionMxDim, transitionMxDim));
  }

  private static class MyNode implements NeuralSpider.Node {
    private final int index;
    private final int wStart;
    private final int wLen;
    private final int pStart;
    private final int pLen;
    private final int nodesCount;
    private final WeightsCalculator calcer;

    private MyNode(int index, int wStart, int wLen, int pStart, int pLen, int nodesCount, WeightsCalculator calcer) {
      this.index = index;
      this.wStart = wStart;
      this.wLen = wLen;
      this.pStart = pStart;
      this.pLen = pLen;
      this.nodesCount = nodesCount;
      this.calcer = calcer;
    }

    public FuncC1 transByParents(final Vec parents) {
      return new FuncC1.Stub() {
        @Override
        public Vec gradientTo(Vec betta, Vec to) {
          final Mx weights = calcer.compute(betta);
          final VecBasedMx grad = new VecBasedMx(pLen - 1, to.sub(wStart, wLen));
          for (int i = 0; i < pLen - 1; i++) {
            final double selectedProbab = weights.get(index, i);
            for (int j = 0; j < pLen - 1; j++) {
              double currentProbab = weights.get(j, i);
              if (j == index)
                grad.set(i, j, parents.get(pStart + i) * selectedProbab * (1 - selectedProbab));
              else
                grad.set(i, j, -parents.get(pStart + i) * selectedProbab * currentProbab);
            }
          }
          return to;
        }

        @Override
        public double value(Vec betta) {
          final Mx weights = calcer.compute(betta);
          return VecTools.multiply(weights.row(index), betta.sub(pStart, pLen - 1));
        }

        @Override
        public int dim() {
          return parents.dim();
        }
      };
    }

    public FuncC1 transByParameters(Vec betta) {
      final Mx weights = calcer.compute(betta);
      return new SubVecFuncC1(new WSum(weights.row(index)), pStart, pLen, nodesCount);
    }
  }
}
