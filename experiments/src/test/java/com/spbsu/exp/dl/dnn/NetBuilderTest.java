package com.spbsu.exp.dl.dnn;

import org.junit.Test;

import com.spbsu.exp.dl.dnn.rectifiers.Flat;
import com.spbsu.exp.dl.dnn.rectifiers.Sigmoid;

import com.spbsu.commons.math.vectors.Mx;

import org.junit.Assert;

import java.io.File;

/**
 * jmll
 *
 * @author ksenon
 */
public class NetBuilderTest extends Assert {

  private static final double DELTA = 1e-14;

  @Test
  public void testBuild() throws Exception {
    final Solver solver = new NetBuilder().buildSolver(new File("src/test/resources/fully_net.json"));

    assertEquals(1000, solver.batchSize);
    assertEquals(123, solver.epochsNumber);
    assertEquals(1e-4, solver.learningRate, DELTA);
    assertEquals(true, solver.debug);

    final FullyConnectedNet net = solver.net;

    assertEquals(true, net.debug);
    checkDim(1000, 10, net.input);
    checkDim(1000, 20, net.output);

    final Layer[] layers = net.layers;

    assertEquals(2, layers.length);

    final Layer first = layers[0];

    assertEquals(0.1, first.bias, DELTA);
    assertEquals(0.0, first.bias_b, DELTA);
    assertEquals(0.0, first.dropoutFraction, DELTA);
    assertEquals(true, first.debug);
    assertTrue(first.rectifier instanceof Sigmoid);
    checkDim(1000, 10, first.input);
    checkDim(1000, 40, first.output);
    checkDim(1000, 40, first.dropoutMask);
    checkDim(1000, 40, first.activations);
    checkDim(40, 10, first.difference);

    final Layer second = layers[1];

    assertEquals(0.4, second.bias, DELTA);
    assertEquals(1.0, second.bias_b, DELTA);
    assertEquals(5.0, second.dropoutFraction, DELTA);
    assertEquals(true, first.debug);
    assertTrue(second.rectifier instanceof Flat);
    checkDim(1000, 40, second.input);
    checkDim(1000, 20, second.output);
    checkDim(1000, 20, second.dropoutMask);
    checkDim(1000, 20, second.activations);
    checkDim(20, 40, second.difference);
  }

  private void checkDim(final int r, final int c, final Mx mx) {
    assertEquals(r, mx.rows());
    assertEquals(c, mx.columns());
  }

}
