package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.models.nn.NeuralSpider;

/**
* User: solar
* Date: 29.06.15
* Time: 17:27
*/
class NFATopology<T> extends NeuralSpider.Topology.Stub {
  private NFANetwork nfaNetwork;
  private final Seq<T> item;
  private final boolean dropout;
  private final NeuralSpider.Node[] nodes;
  private final boolean[] dropoutArr;

  public NFATopology(NFANetwork nfaNetwork, Seq<T> seq, boolean dropout, NeuralSpider.Node[] nodes, boolean[] dropoutArr) {
    this.nfaNetwork = nfaNetwork;
    this.item = seq;
    this.dropout = dropout;
    this.nodes = nodes;
    this.dropoutArr = dropoutArr;
  }

  @Override
  public int outputCount() {
    return NFANetwork.OUTPUT_NODES;
  }

  @Override
  public boolean isDroppedOut(int nodeIndex) {
    if (!dropout || nodeIndex == 0 || nodeIndex == nodes.length - 1)
      return false;
    final int state = nodeIndex % nfaNetwork.statesCount;
    return dropoutArr[state];
  }

  @Override
  public NeuralSpider.Node at(int i) {
    return nodes[i];
  }

  @Override
  public int length() {
    return nodes.length;
  }
}
