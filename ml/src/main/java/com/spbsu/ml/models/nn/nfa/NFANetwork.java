package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.func.generic.SubVecFuncC1;
import com.spbsu.ml.func.generic.WSum;
import com.spbsu.ml.models.nn.NeuralSpider;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 13:25
 */
public class NFANetwork<T> extends NeuralSpider<T, Seq<T>> {
  public static final int OUTPUT_NODES = 2;
  private final FastRandom rng;
  private final double dropout;
  final int statesCount;
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
    final Node[] nodes = new Node[(seq.length() + 1) * statesCount + OUTPUT_NODES];
    for (int i = 0; i < statesCount; i++) {
      final Const aConst = new Const(i == 0 ? 1 : 0);
      nodes[i] = new InputNode(aConst);
    }
    final boolean[] dropoutArr = new boolean[statesCount];

    if (dropout && rng.nextDouble() < this.dropout)
      dropoutArr[rng.nextInt(statesCount - OUTPUT_NODES) + 1] = true;

    final int[] outputNodesConnections = new int[seq.length()];
    for (int d = 0; d < seq.length(); d++) {
      final int elementIndex = ArrayTools.indexOf(seq.at(d), alpha);
      final int prevLayerStart = d * statesCount;
      final WeightsCalculator calcer = calculators.get()[elementIndex];
      calcer.setDropOut(dropoutArr);
      for (int i = 0; i < statesCount; i++) {
        final int nodeIndex = (d + 1) * statesCount + i;
        nodes[nodeIndex] = new MyNode(i, elementIndex * transitionMxDim, transitionMxDim, prevLayerStart, statesCount, nodes.length + statesCount, calcer);
      }
      outputNodesConnections[d] = (d + 2) * statesCount - 1;
    }
    final int lastLayerStart = seq.length() * statesCount;
    nodes[nodes.length - 2] = new NonDeterminedNode(this, lastLayerStart, nodes);
    nodes[nodes.length - 1] = new OutputNode(outputNodesConnections, nodes);

    return new NFATopology<>(this, seq, dropout, nodes, dropoutArr);
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
    for (int i = 0; i <= seq.length(); i++) {
      if (i > 0)
        builder.append(seq.at(i - 1));
      else
        builder.append(" ");

      for (int s = 0; s < statesCount; s++) {
        builder.append("\t").append(CharSeqTools.prettyPrint.format(state.get(i * statesCount + s)));
      }
      builder.append('\n');
    }
    builder.append(" ");
    for (int i = (seq.length() + 1) * statesCount; i < state.length(); i++) {
      builder.append("\t").append(CharSeqTools.prettyPrint.format(state.get(i)));
    }
    builder.append('\n');
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
          final int bettaDim = pLen - 1;
          final int indexLocal = index;
          final int pStartLocal = pStart;
          final VecBasedMx grad = new VecBasedMx(bettaDim, to.sub(wStart, wLen));
          for (int i = 0; i < bettaDim; i++) {
            final double selectedProbab = weights.get(indexLocal, i);
            for (int j = 0; j < bettaDim; j++) {
              double currentProbab = weights.get(j, i);
              if (j == indexLocal)
                grad.set(i, j, parents.get(pStartLocal + i) * selectedProbab * (1 - selectedProbab));
              else
                grad.set(i, j, -parents.get(pStartLocal + i) * selectedProbab * currentProbab);
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
