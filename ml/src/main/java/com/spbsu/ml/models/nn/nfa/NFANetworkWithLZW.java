package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.models.nn.NeuralSpider;

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
    for (int d = 0; d < seq.length(); d++) {
      final int elementIndex = dictionary.get().indexOf(seq.sub(d, d+1));
      final int prevLayerStart = d * statesCount;
      final WeightsCalculator calcer = calculators.get().get(elementIndex);
      calcer.setDropOut(dropoutArr);
      for (int i = 0; i < statesCount; i++) {
        final int nodeIndex = (d + 1) * statesCount + i;
        nodes[nodeIndex] = new MyNode(i, elementIndex * transitionMxDim, transitionMxDim, dim, prevLayerStart, statesCount, nodes.length, calcer);
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
}
