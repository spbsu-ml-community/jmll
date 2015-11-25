package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.func.generic.SubVecFuncC1;
import com.spbsu.ml.func.generic.WSum;
import com.spbsu.ml.models.nn.NeuralSpider;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.spbsu.commons.math.vectors.VecTools.append;
import static com.spbsu.commons.math.vectors.VecTools.fill;
import static com.spbsu.commons.math.vectors.VecTools.scale;

/**
 * Created by afonin.s on 06.11.2015.
 */
public class NFANetworkWithLZW<T> extends NeuralSpider<T, Seq<T>> {
  private final FastRandom rng;
  private final double dropout;
  final int statesCount;
  private final Seq<T> alpha;
  private final int dim;
  private final int transitionMxDim;
  private final int finalStates;
  private final ThreadLocal<Map<Seq<T>, SeqWeightsCalculator>> dictionary;


  public NFANetworkWithLZW(FastRandom rng, double dropout, int statesCount, int finalStates, Seq<T> alpha) {
    super();
    this.rng = rng;
    this.dropout = dropout;
    this.statesCount = statesCount;
    this.finalStates = finalStates;
    this.alpha = alpha;
    transitionMxDim = (statesCount - finalStates) * (statesCount - 1);
    dim = transitionMxDim * alpha.length();
    dictionary = new ThreadLocal<Map<Seq<T>, SeqWeightsCalculator>>() {
      @Override
      protected Map<Seq<T>, SeqWeightsCalculator> initialValue() {
        final Map<Seq<T>, SeqWeightsCalculator> result = new HashMap<>(alpha.length());
        for (int i = 0; i < alpha.length(); i++) {
          result.put(alpha.sub(i, i + 1), new SeqWeightsCalculator(statesCount, finalStates, transitionMxDim, i * transitionMxDim));
        }
        return result;
      }
    };
  }

  @Override
  protected Topology topology(Seq<T> seq, boolean dropout) {
//    final Node[] nodes = new Node[(seq.length() + 1) * statesCount + finalStates + 1];
    final int nodesCount = (seq.length() + 1) * statesCount + finalStates + 1;
    final List<Node> nodes = new ArrayList<>();
    for (int i = 0; i < statesCount; i++) {
      final Const aConst = new Const(i == 0 ? 1 : 0);
      nodes.add(new InputNode(aConst));
    }
    final boolean[] dropoutArr = new boolean[statesCount];

    if (dropout && rng.nextDouble() < this.dropout)
      dropoutArr[rng.nextInt(statesCount - finalStates - 1) + 1] = true;

    final int[][] outputNodesConnections = new int[finalStates][seq.length()];
    int start = 0, end = 0;
    for (int d = 0; d < seq.length() + 1; d++) {
      if (end != seq.length())
        end++;
      Seq<T> sub = seq.sub(start, end);
      SeqWeightsCalculator calcer = dictionary.get().get(sub);
      if (calcer == null) {
        final int subLength = sub.length();
        final int[] wStarts = new int[subLength];
        for (int i = 0; i < subLength; i++) {
          wStarts[i] = SeqTools.indexOf(alpha, sub.at(i)) * transitionMxDim;
        }
        dictionary.get().put(sub, new SeqWeightsCalculator(statesCount, finalStates, transitionMxDim, wStarts));

        final Seq<T> prevSub = sub.sub(0, subLength - 1);
        final SeqWeightsCalculator prevCalcer = dictionary.get().get(prevSub);
        fillLayer(nodes, outputNodesConnections, d, prevSub, prevCalcer, dropoutArr, nodesCount);

        start += prevSub.length();
      } else if (end == seq.length()) {
        fillLayer(nodes, outputNodesConnections, d, sub, calcer, dropoutArr, nodesCount);
      }
    }
    final int lastLayerStart = seq.length() * statesCount;
    nodes.add(new NonDeterminedNode(statesCount - finalStates, lastLayerStart, nodesCount));
    for (int i = 0; i < finalStates; i++) {
      nodes.add(new OutputNode(outputNodesConnections[i], nodes.toArray(new Node[] {})));
    }

    return new NFATopology<>(statesCount, finalStates, dropout, nodes.toArray(new Node[] {}), dropoutArr);
  }

  private void fillLayer(List<Node> nodes, int[][] outputNodesConnections, int d, Seq<T> sub, SeqWeightsCalculator calcer, boolean[] dropoutArr, int nodesCount) {
    final int prevLayerStart = (d - 1) * statesCount;
    calcer.setDropOut(dropoutArr);
    for (int i = 0; i < statesCount; i++) {
      //final int nodeIndex = d * statesCount + i;
      nodes.add(new MyNode(i, transitionMxDim, dim, prevLayerStart, statesCount, nodesCount, calcer, sub));
    }
    for (int i = 0; i < finalStates; i++) {
      outputNodesConnections[i][d - 1] = (d + 1) * statesCount - finalStates + i;
    }
  }

  @Override
  public int dim() {
    return dim;
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

  protected class MyNode implements NeuralSpider.Node {
    private final int index;
    private final int wLen;
    private final int wDim;
    private final int pStart;
    private final int pLen;
    private final int nodesCount;
    private final SeqWeightsCalculator calcer;
    private final Seq<T> seq;

    public MyNode(int index, int wLen, int wDim, int pStart, int pLen, int nodesCount, SeqWeightsCalculator calcer, Seq<T> seq) {
      this.index = index;
      this.wLen = wLen;
      this.wDim = wDim;
      this.pStart = pStart;
      this.pLen = pLen;
      this.nodesCount = nodesCount;
      this.calcer = calcer;
      this.seq = seq;
    }

    @Override
    public FuncC1 transByParents(Vec parents) {
      return new FuncC1.Stub() {

        private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

        @Override
        public Vec gradientTo(Vec betta, Vec to) {
          final int[] wStarts = calcer.getwStarts();
          final int length = wStarts.length - 1;
          final Node[] silentNodes = new Node[length * pLen + 1];

          final boolean[] dropoutArr = new boolean[statesCount];
          if (rng.nextDouble() < NFANetworkWithLZW.this.dropout)
            dropoutArr[rng.nextInt(statesCount - finalStates - 1) + 1] = true;

          for (int i = 0; i < length; i++) {
            WeightsCalculator calculator = NFANetworkWithLZW.this.dictionary.get().get(seq.sub(i, i + 1));
            calculator.setDropOut(dropoutArr);
            for (int j = 0; j < pLen; j++) {
              final int nodeIndex = i * statesCount + j;
              silentNodes[nodeIndex] = new NFANetwork.MyNode(j, wStarts[i], wLen, wDim, pStart + i * pLen, pLen, nodesCount, calculator);
            }
          }
          WeightsCalculator calculator = NFANetworkWithLZW.this.dictionary.get().get(seq.sub(length, length + 1));
          calculator.setDropOut(dropoutArr);
          silentNodes[silentNodes.length - 1] = new NFANetwork.MyNode(index, wStarts[length], wLen, wDim, pStart + length * pLen, pLen, nodesCount, calculator);

          for (int i = 0; i < silentNodes.length - 1; i++) {
            Node silentNode = silentNodes[i];
            final double value = silentNode.transByParameters(betta).value(parents);
            parents.set(pStart + pLen + i, value);
          }


          final Vec dT = gradientSCache.get(nodesCount);
          dT.set(pStart + silentNodes.length - 1, 1.0);
          final Vec prevLayerGrad = new SparseVec(nodesCount);
          final Vec wGrad = new SparseVec(wDim);
          for (int nodeIndex = silentNodes.length - 1; nodeIndex >= 0; nodeIndex--) {
            final double dTds_i = dT.get(nodeIndex + pStart);

            final Node node = silentNodes[nodeIndex];
            fill(prevLayerGrad, 0.);
            fill(wGrad, 0.);

            node.transByParameters(betta).gradientTo(parents, prevLayerGrad);
            node.transByParents(parents).gradientTo(betta, wGrad);

            scale(prevLayerGrad, dTds_i);
            scale(wGrad, dTds_i);

            append(dT, prevLayerGrad);
            append(to, wGrad);
          }
          return to;
        }

        @Override
        public double value(Vec betta) {
          final Mx weights = calcer.compute(betta);
          return VecTools.multiply(weights.row(index), parents.sub(pStart, pLen));
        }

        @Override
        public int dim() {
          return wDim;
        }
      };
    }

    public FuncC1 transByParameters(Vec betta) {
      final Mx weights = calcer.compute(betta);
      return new SubVecFuncC1(new WSum(weights.row(index)), pStart, pLen, nodesCount);
    }
  }
}
