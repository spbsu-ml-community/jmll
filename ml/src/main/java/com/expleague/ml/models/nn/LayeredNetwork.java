package com.expleague.ml.models.nn;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.layers.FCCalcer;

import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
* User: solar
* Date: 26.05.15
* Time: 11:46
*/
public class LayeredNetwork extends NeuralSpider<Double, Vec> {
  private final NodeCalcer[] nodeCalcers;
  private final Random rng;
  private final double dropout;
  private final int[] config;
  private final int dim;
  private final int numParameters;

  public LayeredNetwork(Random rng, double dropout, final int... config) {
    this.rng = rng;
    this.dropout = dropout;
    this.config = config;

    this.dim = config[0];
    int len = IntStream.of(config).sum();
    this.nodeCalcers = new NodeCalcer[len];
    int layer = 0;
    int index = 0;
    int wCount = 0;
    NodeCalcer current = new StartNodeCalcer();
    for (int i = 0; i < len; i++) {
      this.nodeCalcers[i] = current;
      if (++index >= config[layer]) {
        layer++;
        if (layer >= config.length)
          break;

        current = new FCCalcer(i + 1, config[layer], i - config[layer - 1] + 1, config[layer - 1], wCount, config[layer - 1] * config[layer]);
        wCount += config[layer - 1] * config[layer];
        index = 0;
      }
    }
    this.numParameters = wCount;
  }

  @Override
  public int xdim() {
    return config[0];
  }

  @Override
  public int numParameters() {
    return numParameters;
  }

  @Override
  protected Topology topology(final boolean dropout) {
    return new Topology.Stub() {
      @Override
      public int outputCount() {
        return config[config.length - 1];
      }

      @Override
      public boolean isDroppedOut(int nodeIndex) {
        //noinspection SimplifiableIfStatement
        if (!dropout || nodeIndex > nodeCalcers.length)
          return false;
        return LayeredNetwork.this.dropout > MathTools.EPSILON && rng.nextDouble() < LayeredNetwork.this.dropout;
      }

      @Override
      public NodeCalcer at(int i) {
        return nodeCalcers[i];
      }

      @Override
      public int dim() {
        return dim;
      }

      @Override
      public int length() {
        return nodeCalcers.length;
      }

      @Override
      public Stream<NodeCalcer> stream() {
        return Stream.of(nodeCalcers);
      }
    };
  }

  private static class StartNodeCalcer implements NeuralSpider.NodeCalcer {
    @Override
    public double apply(Vec state, Vec betta, int nodeIdx) {
      return state.get(nodeIdx);
    }

    @Override
    public int start(int nodeIdx) {
      return 0;
    }

    @Override
    public int end(int nodeIdx) {
      return 0;
    }

    @Override
    public void gradByStateTo(Vec state, Vec betta, Vec to) {
    }

    @Override
    public void gradByParametersTo(Vec state, Vec betta, Vec to) {
    }
  }
}
