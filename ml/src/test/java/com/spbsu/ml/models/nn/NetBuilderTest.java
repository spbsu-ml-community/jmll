package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NetBuilderTest {
  private static class Identity extends AnalyticFunc.Stub {
    @Override
    public double value(double x) {
      return x;
    }

    @Override
    public double gradient(double x) {
      return 1.;
    }
  }

  @Test
  public void testConvolution1x1() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .conv2D(1, 1, 1, 1, 1,
        new Identity(), "conv1", NetBuilder.DATA_LAYER_NAME).build();
    net.initialize(weights -> VecTools.fill(weights, 1.));

    double[] data = {
        1, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        3, 6, 9, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    Vec input = new ArrayVec(data);
    assertEquals(input, net.forward(input));

    net.initialize(weights -> VecTools.fill(weights, 5.));

    Vec input5x = VecTools.copy(input);
    VecTools.scale(input5x, 5.);
    assertEquals(input5x, net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .conv2D(1, 1, 2, 2, 1, new Identity(), "conv1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride2 = {
        1, 3, 5,
        3, 9, 2,
        6, 3, 2
    };

    Vec vecOutStride2 = new ArrayVec(outStride2);
    net.initialize(weights -> VecTools.fill(weights, 1.));
    assertEquals(vecOutStride2, net.forward(input));

    VecTools.scale(vecOutStride2, 5.);
    net.initialize(weights -> VecTools.fill(weights, 5.));
    assertEquals(vecOutStride2, net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .conv2D(1, 1, 4, 4, 1, new Identity(), "conv1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride4 = {
        1, 5,
        6, 2
    };

    Vec vecOutStride4 = new ArrayVec(outStride4);
    net.initialize(weights -> VecTools.fill(weights, 1.));
    assertEquals(vecOutStride4, net.forward(input));

    VecTools.scale(vecOutStride4, 5.);
    net.initialize(weights -> VecTools.fill(weights, 5.));
    assertEquals(vecOutStride4, net.forward(input));
  }

  @Test
  public void testConvolution3x3() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .conv2D(3, 3, 1, 1, 1,
        new Identity(), "conv1", NetBuilder.DATA_LAYER_NAME)
        .build();
    net.initialize(weights -> VecTools.fill(weights, 1.));

    double[] data = {
        1, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        3, 6, 9, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    double[] out = {
        30, 37, 42,
        37, 42, 37,
        42, 39, 28
    };

    Vec input = new ArrayVec(data);
    Vec outVec = new ArrayVec(out);
    assertEquals(outVec, net.forward(input));

    net.initialize(weights -> VecTools.fill(weights, 5.));

    Vec outVec5x = VecTools.copy(outVec);
    VecTools.scale(outVec5x, 5.);
    assertEquals(outVec5x, net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .conv2D(3, 3, 1, 2, 1, new Identity(), "conv1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride2 = {
        30, 42,
        37, 37,
        42, 28
    };

    Vec vecOutStride2 = new ArrayVec(outStride2);
    net.initialize(weights -> VecTools.fill(weights, 1.));
    assertEquals(vecOutStride2, net.forward(input));

    VecTools.scale(vecOutStride2, 5.);
    net.initialize(weights -> VecTools.fill(weights, 5.));
    assertEquals(vecOutStride2, net.forward(input));
  }

  @Test
  public void testConvolution5x5() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .conv2D(5, 5, 1, 1, 1,
            new Identity(), "conv1", NetBuilder.DATA_LAYER_NAME)
        .build();
    net.initialize(weights -> VecTools.fill(weights, 1.));

    double[] data = {
        1, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        3, 6, 9, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    double[] out = {
        92
    };

    Vec input = new ArrayVec(data);
    Vec outVec = new ArrayVec(out);
    assertEquals(outVec, net.forward(input));

    net.initialize(weights -> VecTools.fill(weights, 5.));

    Vec outVec5x = VecTools.copy(outVec);
    VecTools.scale(outVec5x, 5.);
    assertEquals(outVec5x, net.forward(input));
  }

  @Test
  public void testPooling1x1() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .maxPool2D(1, 1, 1, 1, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] data = {
        1, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        3, 6, 9, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    Vec input = new ArrayVec(data);
    assertEquals(input, net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(1, 1, 1, 2, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride1x2 = {
        1, 3, 5,
        0, 4, 8,
        3, 9, 2,
        4, 1, 1,
        6, 3, 2
    };

    assertEquals(new ArrayVec(outStride1x2), net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(1, 1, 2, 1, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride2x1 = {
        1, 2, 3, 4, 5,
        3, 6, 9, 1, 2,
        6, 2, 3, 4, 2
    };

    assertEquals(new ArrayVec(outStride2x1), net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(1, 1, 2, 2, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride2x2 = {
        1, 3, 5,
        3, 9, 2,
        6, 3, 2
    };

    assertEquals(new ArrayVec(outStride2x2), net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(1, 1, 4, 4, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] outStride4x4 = {
        1, 5,
        6, 2
    };

    assertEquals(new ArrayVec(outStride4x4), net.forward(input));
  }

  @Test
  public void testPooling2x2() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .maxPool2D(2, 2, 1, 1, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] data = {
        1, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        3, 6, 9, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    double[] out = {
        2, 4, 6, 8,
        6, 9, 9, 8,
        8, 9, 9, 5,
        8, 8, 5, 5
    };

    Vec input = new ArrayVec(data);
    Vec outVec = new ArrayVec(out);
    assertEquals(outVec, net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(2, 2, 2, 2, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] out2x2 = {
        2, 6,
        8, 9
    };

    assertEquals(new ArrayVec(out2x2), net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(2, 2, 3, 3, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] out3x3 = {
        2, 8,
        8, 5
    };

    assertEquals(new ArrayVec(out3x3), net.forward(input));
  }

  @Test
  public void testPooling3x3() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .maxPool2D(3, 3, 1, 1, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] data = {
        9, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        1, 6, 3, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    double[] out = {
        9, 6, 8,
        8, 8, 8,
        8, 8, 5
    };

    Vec input = new ArrayVec(data);
    Vec outVec = new ArrayVec(out);
    assertEquals(outVec, net.forward(input));

    net = new NetBuilder(1, 5, 5)
        .maxPool2D(3, 3, 2, 2, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] out2x2 = {
        9, 8,
        8, 5
    };

    assertEquals(new ArrayVec(out2x2), net.forward(input));
  }

  @Test
  public void testPooling5x5() {
    ConvNet net = new NetBuilder(1, 5, 5)
        .maxPool2D(5, 5, 1, 1, "pool1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] data = {
        9, 2, 3, 4, 5,
        0, 2, 4, 6, 8,
        1, 6, 3, 1, 2,
        4, 8, 1, 5, 1,
        6, 2, 3, 4, 2
    };

    double[] out = {
        9
    };

    Vec input = new ArrayVec(data);
    Vec outVec = new ArrayVec(out);
    assertEquals(outVec, net.forward(input));
  }

  @Test
  public void testDense() {
    ConvNet net = new NetBuilder(10)
        .dense(1, new Identity(), "fc1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] data = { 0, 2, 8, 3, 1, 2, 3, 4, 9, 7 };

    ArrayVec input = new ArrayVec(data);
    net.initialize(weights -> VecTools.fill(weights, 1.));

    double[] out = { 39 };

    ArrayVec output = new ArrayVec(out);
    assertEquals(output, net.forward(input));

    net = new NetBuilder(10)
        .dense(10, new Identity(), "fc1", NetBuilder.DATA_LAYER_NAME)
        .build();

    double[] out10 = new double[10];
    for (int i = 0; i < out10.length; i++) {
      out10[i] = out[0] * (i + 1);
    }

    net.initialize(weights -> {
      for (int i = 0; i < 100; i++) {
        weights.set(i, i / 10 + 1);
      }
    });

    assertEquals(new ArrayVec(out10), net.forward(input));
  }
}
