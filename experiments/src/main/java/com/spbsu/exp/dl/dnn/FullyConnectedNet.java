package com.spbsu.exp.dl.dnn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.logging.Logger;

/**
 * jmll
 *
 * @author ksenon
 */
public class FullyConnectedNet {

  private static final Logger LOG = Logger.create(FullyConnectedNet.class);

  public Layer[] layers;

  public Mx input;
  public Mx output;

  public boolean debug;

  public void forward() {
    layers[0].input = input;

    for (int i = 0; i < layers.length - 1; i++) {
      if (debug) {
        LOG.info("Forward layer #" + i);
      }

      layers[i].forward();
      layers[i + 1].input = layers[i].output;
    }

    layers[layers.length - 1].forward();
    output = layers[layers.length - 1].output;

    if (debug) {
      LOG.info("Forward layer #" + (layers.length - 1));
    }
  }

  public void backward(final Mx error) {
    final Mx scale = VecTools.scale(error, -1);
    final Layer lastLayer = layers[layers.length - 1];

    lastLayer.rectifier.grad(lastLayer.activations, lastLayer.output);
    for (int i = 0; i < scale.dim(); i++) {
      lastLayer.output.set(i, lastLayer.output.get(i) * scale.get(i));
    }

    for (int i = layers.length - 1; i > 0; i--) {
      layers[i].backward();
      layers[i - 1].output = layers[i].input;
    }
    layers[0].backward();
  }

}
