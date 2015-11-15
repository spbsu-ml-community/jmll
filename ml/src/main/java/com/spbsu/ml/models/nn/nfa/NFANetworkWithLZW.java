package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.models.nn.NeuralSpider;

import static com.spbsu.commons.math.vectors.VecTools.append;
import static com.spbsu.commons.math.vectors.VecTools.fill;
import static com.spbsu.commons.math.vectors.VecTools.scale;

/**
 * Created by afonin.s on 06.11.2015.
 */
public class NFANetworkWithLZW<T> extends NFANetwork<T> {

  public NFANetworkWithLZW(FastRandom rng, double dropout, int statesCount, int finalStates, Seq<T> alpha) {
    super(rng, dropout, statesCount, finalStates, alpha);
  }

  @Override
  protected Topology topology(Seq<T> seq, boolean dropout) {
    final Node[] nodes = new Node[(seq.length() + 1) * statesCount + finalStates + 1];
    for (int i = 0; i < statesCount; i++) {
      final Const aConst = new Const(i == 0 ? 1 : 0);
      nodes[i] = new InputNode(aConst);
    }
    final boolean[] dropoutArr = new boolean[statesCount];

    if (dropout && rng.nextDouble() < this.dropout)
      dropoutArr[rng.nextInt(statesCount - finalStates - 1) + 1] = true;

    final int[][] outputNodesConnections = new int[finalStates][seq.length()];
    int start = 0, end = 0;
    for (int d = 0; d < seq.length(); d++) {
      end++;
      Seq<T> sub = seq.sub(start, end);
      int elementIndex = dictionary.get().indexOf(sub);
      if (elementIndex < 0) {
        dictionary.get().add(sub);
        elementIndex = dictionary.get().indexOf(seq.sub(start, end - 1));
      }
//      final int prevLayerStart = d * statesCount;

//      if (calculators.get().size() <= elementIndex) {
//        calculators.
//      }
      final WeightsCalculator calcer = calculators.get().get(elementIndex);
      calcer.setDropOut(dropoutArr);
      for (int i = 0; i < statesCount; i++) {
        final int nodeIndex = (d + 1) * statesCount + i;
//        nodes[nodeIndex] = new MyNode(i, elementIndex * transitionMxDim, transitionMxDim, dim, prevLayerStart, statesCount, nodes.length, calcer);
      }
      for (int i = 0; i < finalStates; i++) {
        outputNodesConnections[i][d] = (d + 2) * statesCount - finalStates + i;
      }
    }
    final int lastLayerStart = seq.length() * statesCount;
    nodes[nodes.length - finalStates - 1] = new NonDeterminedNode(statesCount - finalStates, lastLayerStart, nodes.length);
    for (int i = 0; i < finalStates; i++) {
      nodes[nodes.length - finalStates + i] = new OutputNode(outputNodesConnections[i], nodes);
    }

    return new NFATopology<>(statesCount, finalStates, dropout, nodes, dropoutArr);
  }

  protected class MyNode extends NFANetwork.MyNode {

    private final int[] wStarts;
    private final int[] elementIndexes;

    public MyNode(int index, int wLen, int wDim, int pStart, int pLen, int nodesCount, WeightsCalculator calcer, int[] wStarts, int[] elementIndexes) {
      super(index, wStarts[0], wLen, wDim, pStart, pLen, nodesCount, calcer);
      if (wStarts.length < 2)
        throw new IllegalArgumentException("This node can't be used for only one layer. Use NFANetwork.MyNode instead of it.");
      this.wStarts = wStarts;
      this.elementIndexes = elementIndexes;
    }

    @Override
    public FuncC1 transByParents(Vec parents) {
      return new FuncC1.Stub() {

        private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();
        @Override
        public Vec gradientTo(Vec betta, Vec to) {
          int length = wStarts.length - 1;
          Node[] silentNodes = new Node[length * pLen + 1];

          final boolean[] dropoutArr = new boolean[statesCount];
          if (rng.nextDouble() < NFANetworkWithLZW.this.dropout)
            dropoutArr[rng.nextInt(statesCount - finalStates - 1) + 1] = true;

          for (int i = 0; i < length; i++) {
            WeightsCalculator calculator = NFANetworkWithLZW.this.calculators.get().get(elementIndexes[i]);
            calculator.setDropOut(dropoutArr);
            for (int j = 0; j < pLen; j++) {
              final int nodeIndex = i * statesCount + j;
              silentNodes[nodeIndex] = new NFANetwork.MyNode(j, wStarts[i], wLen, wDim, pStart + i * pLen, pLen, nodesCount, calculator);
            }
          }
          WeightsCalculator calculator = NFANetworkWithLZW.this.calculators.get().get(elementIndexes[length]);
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
  }
}
