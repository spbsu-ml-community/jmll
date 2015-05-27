package com.spbsu.exp.dl.dnn;

import org.junit.Test;

import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.exp.dl.dnn.rectifiers.Sigmoid;
import org.junit.Assert;

/**
 * jmll
 *
 * @author ksenon
 */
public class LayerTest extends Assert {

  private static final double DELTA = 1e-13;

  @Test
  public void testForward() throws Exception {
    final Layer layer = new Layer();
    layer.isTrain = true;
    layer.dropoutFraction = 2;
    layer.rectifier = new Sigmoid();
    layer.weights = new VecBasedMx(4, 3);
    layer.activations = new VecBasedMx(2, 4);

    layer.input = new VecBasedMx(3, new ArrayVec(
        -10, 0, 10,
        -5, -1, 1
    ));
    layer.output = new VecBasedMx(2, 4);

    layer.forward();

    for (int i = 0; i < layer.output.dim(); i++) {
      assertEquals(0., layer.output.get(i), DELTA);
    }

    layer.dropoutFraction = 0;

    layer.forward();

    for (int i = 0; i < layer.output.dim(); i++) {
      assertEquals(0.5, layer.output.get(i), DELTA);
    }
  }

  @Test
  public void testBackward() throws Exception {
    final Layer layer = new Layer();
    layer.isTrain = true;
    layer.dropoutFraction = 2;
    layer.rectifier = new Sigmoid();
    layer.weights = new VecBasedMx(4, 3);
    layer.activations = new VecBasedMx(2, 4);
    layer.dropoutMask = new VecBasedMx(3, new ArrayVec(
        1, 1, 1, 1, 1, 1
    ));

    layer.input = new VecBasedMx(2, 3);
    layer.output = new VecBasedMx(4, new ArrayVec(
        -10, 0, 10, 100,
        -5, -1, 1, 10
    ));

    layer.backward();

    for (int i = 0; i < layer.input.dim(); i++) {
      assertEquals(0., layer.input.get(i), DELTA);
    }

    layer.dropoutFraction = 0;

    layer.backward();

    for (int i = 0; i < layer.input.dim(); i++) {
      assertEquals(0., layer.input.get(i), DELTA);
    }
  }

}
