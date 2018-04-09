package com.expleague.ml.models.nn;

import com.expleague.ml.func.generic.ReLU;
import com.expleague.ml.models.nn.nodes.FCCalcer;
import com.expleague.ml.models.nn.nodes.InputNodeCalcer;

import java.util.Random;
import java.util.stream.IntStream;

import static com.expleague.ml.models.nn.NeuralSpider.*;

/**
* User: solar
* Date: 26.05.15
* Time: 11:46
*/
public class LayeredNetwork {
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
    NodeCalcer current = new InputNodeCalcer();
    for (int i = 0; i < len; i++) {
      this.nodeCalcers[i] = current;
      if (++index >= config[layer]) {
        layer++;
        if (layer >= config.length)
          break;

        current = new FCCalcer(i + 1, config[layer],
            i - config[layer - 1] + 1, config[layer - 1],
            wCount, config[layer - 1] * config[layer], new ReLU());
        wCount += config[layer - 1] * config[layer];
        index = 0;
      }
    }
    this.numParameters = wCount;
  }
}
