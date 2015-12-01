package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.func.generic.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
* User: solar
* Date: 26.05.15
* Time: 11:46
*/
public class LayeredNetwork extends NeuralSpider<Double, Vec> {
  private final Node[] nodes;
  private final Random rng;
  private final double dropout;
  private final int[] config;
  private final int dim;

  public LayeredNetwork(Random rng, double dropout, final int... config) {
    this.rng = rng;
    this.dropout = dropout;
    this.config = config;
    final int nodesCount;
    {
      int dim = config[0];
      int count = 1 + config[0];
      for (int i = 1; i < config.length; i++) {
        count += config[i];
        dim += config[i] * config[i - 1];
      }
      this.dim = dim;
      nodesCount = count;
    }
    int wStart = config[0];
    final List<Node> nodes = new ArrayList<>();
    for(int d = 1; d < config.length; d++) {
      final int prevLayerPower = config[d - 1];
      final int layerStart = nodes.size() + config[0] + 1;

      for (int i = 0; i < config[d]; i++) {
        final int fwStart = wStart;
        nodes.add(new Node() {
          @Override
          public FuncC1 transByParameters(Vec betta) {
            return new SubVecFuncC1(new WSumSigmoid(betta.sub(fwStart, prevLayerPower)), layerStart - prevLayerPower, prevLayerPower, nodesCount);
          }

          @Override
          public FuncC1 transByParents(Vec state) {
            return new SubVecFuncC1(new WSumSigmoid(state.sub(layerStart - prevLayerPower, prevLayerPower)), fwStart, prevLayerPower, dim);
          }
        });
        wStart += prevLayerPower;
      }
    }
    this.nodes = nodes.toArray(new Node[nodes.size()]);
  }
  @Override
  public int dim() {
    return dim;
  }

  @Override
  protected Topology topology(final Vec argument, final boolean dropout) {
    if (argument.dim() != config[0])
      throw new IllegalArgumentException();
    final Node[] inputLayer = new Node[config[0]];
    for(int i = 0; i < inputLayer.length; i++) {
      final int nindex = i;
      inputLayer[i] = new Node() {
        @Override
        public FuncC1 transByParameters(Vec betta) {
          return new Const(argument.get(nindex));
        }

        @Override
        public FuncC1 transByParents(Vec state) {
          return new Const(argument.get(nindex));
        }
      };
    }

    return new Topology.Stub() {
      @Override
      public int outputCount() {
        return config[config.length - 1];
      }

      @Override
      public boolean isDroppedOut(int nodeIndex) {
        //noinspection SimplifiableIfStatement
        if (!dropout || nodeIndex < inputLayer.length || nodeIndex > nodes.length - inputLayer.length)
          return false;
        return LayeredNetwork.this.dropout > MathTools.EPSILON && rng.nextDouble() < LayeredNetwork.this.dropout;
      }

      @Override
      public Node at(int i) {
        return i <= inputLayer.length ? inputLayer[i - 1] : nodes[i - inputLayer.length - 1];
      }

      @Override
      public int length() {
        return inputLayer.length + nodes.length + 1;
      }
    };
  }
}
