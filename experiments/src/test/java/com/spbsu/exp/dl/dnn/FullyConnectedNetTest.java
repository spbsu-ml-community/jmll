package com.spbsu.exp.dl.dnn;

import org.junit.Test;

import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.exp.dl.dnn.rectifiers.Sigmoid;

import org.junit.Assert;

/**
 * jmll
 *
 * @author ksenon
 */
public class FullyConnectedNetTest extends Assert {

  private static final double DELTA = 1e-13;

  @Test
  public void testForward() throws Exception {
    final Layer layer_1 = new Layer();
    layer_1.isTrain = true;
    layer_1.dropoutFraction = 0;
    layer_1.rectifier = new Sigmoid();
    layer_1.weights = new VecBasedMx(4, 3);
    layer_1.activations = new VecBasedMx(2, 4);
    layer_1.input = new VecBasedMx(2, 3);
    layer_1.output = new VecBasedMx(2, 4);

    final Layer layer_2 = new Layer();
    layer_2.isTrain = true;
    layer_2.dropoutFraction = 0;
    layer_2.rectifier = new Sigmoid();
    layer_2.weights = new VecBasedMx(5, 4);
    layer_2.activations = new VecBasedMx(2, 5);
    layer_2.input = new VecBasedMx(2, 4);
    layer_2.output = new VecBasedMx(2, 5);

    final Layer layer_3 = new Layer();
    layer_3.isTrain = true;
    layer_3.dropoutFraction = 0;
    layer_3.rectifier = new Sigmoid();
    layer_3.weights = new VecBasedMx(1, 5);
    layer_3.activations = new VecBasedMx(2, 1);
    layer_3.input = new VecBasedMx(2, 5);
    layer_3.output = new VecBasedMx(2, 1);

    final FullyConnectedNet net = new FullyConnectedNet();
    net.layers = new Layer[]{layer_1, layer_2, layer_3};
    net.input = new VecBasedMx(2, 3);
    net.output = new VecBasedMx(2, 1);

    net.forward();

    for (int i = 0; i < net.output.length(); i++) {
      assertEquals(0.5, net.output.get(i), DELTA);
    }
  }

  @Test
  public void testBackward() throws Exception {

    final Layer layer_1 = new Layer();
    layer_1.isTrain = true;
    layer_1.dropoutFraction = 0;
    layer_1.rectifier = new Sigmoid();
    layer_1.weights = new VecBasedMx(4, 3);
    layer_1.activations = new VecBasedMx(2, 4);
    layer_1.input = new VecBasedMx(2, 3);
    layer_1.output = new VecBasedMx(2, 4);

    final Layer layer_2 = new Layer();
    layer_2.isTrain = true;
    layer_2.dropoutFraction = 0;
    layer_2.rectifier = new Sigmoid();
    layer_2.weights = new VecBasedMx(5, 4);
    layer_2.activations = new VecBasedMx(2, 5);
    layer_2.input = new VecBasedMx(2, 4);
    layer_2.output = new VecBasedMx(2, 5);

    final Layer layer_3 = new Layer();
    layer_3.isTrain = true;
    layer_3.dropoutFraction = 0;
    layer_3.rectifier = new Sigmoid();
    layer_3.weights = new VecBasedMx(1, 5);
    layer_3.activations = new VecBasedMx(2, 1);
    layer_3.input = new VecBasedMx(2, 5);
    layer_3.output = new VecBasedMx(2, 1);

    final FullyConnectedNet net = new FullyConnectedNet();
    net.layers = new Layer[]{layer_1, layer_2, layer_3};
    net.input = new VecBasedMx(2, 3);
    net.output = new VecBasedMx(2, 1);

    net.forward();
    net.backward(new VecBasedMx(2, 1));

    for (int i = 0; i < net.output.length(); i++) {
      assertEquals(0., net.output.get(i), DELTA);
    }
  }

}
