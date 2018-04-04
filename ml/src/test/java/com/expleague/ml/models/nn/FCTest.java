package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.models.nn.layers.ConstSizeInput;
import com.expleague.ml.models.nn.layers.FCLayerBuilder;
import com.expleague.ml.models.nn.layers.OneOutLayer;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;

public class FCTest {
  private static final NeuralSpider<Vec> spider = new NeuralSpider<>();
  private static final int ROUNDS = 100;
  private static final FastRandom rng = new FastRandom();

  @Test
  public void testOneLayer() {
    for (int nIn = 1; nIn <= 100; nIn += 5) {
      for (int nOut = 1; nOut <= 100; nOut += 5) {
        System.out.println("Test [" + nIn + ", " + nOut + "]");
        final NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
            new ConstSizeInput(nIn))
            .append(new FCLayerBuilder().nOut(nOut))
            .build(new OneOutLayer());

        testNN(network, nIn, nOut);
      }
    }
  }

  @Test
  public void testMultiLayer() {
    for (int i = 0; i < 100; i++) {
      final int numLayers = rng.nextInt(20) + 1;
      final int[] dims = new int[numLayers];
      generateDims(dims);

      System.out.print("Test [");
      for (int j = 0; j < dims.length - 1; j++) {
        System.out.print(dims[j] + ", ");
      }
      System.out.println(dims[dims.length - 1] + "]");

      NetworkBuilder<Vec> builder = new NetworkBuilder<>(new ConstSizeInput(dims[0]));
      for (int j = 1; j < numLayers; j++) {
        builder.append(new FCLayerBuilder().nOut(dims[j]));
      }
      final NetworkBuilder<Vec>.Network network = builder.build(new OneOutLayer());

      testNN(network, dims);
    }
  }

  private void generateDims(int[] dims) {
    for (int i = 0; i < dims.length; i++) {
      dims[i] = rng.nextInt(100) + 1;
    }
  }

  private void testNN(NetworkBuilder<Vec>.Network network, int... dims) {
    for (int i = 0; i < ROUNDS; i++) {
      Vec weights = new ArrayVec(network.wdim());
      VecTools.fillUniform(weights, rng);

      Vec arg = new ArrayVec(dims[0]);
      VecTools.fillUniform(arg, rng);

      Vec result = spider.compute(network, arg, weights);
      Vec expected = matmulNN(arg, weights, dims);
      assertTrue(VecTools.equals(expected, result, 1e-6));
    }
  }

  private Vec matmulNN(Vec arg, Vec weights, int... dims) {
    Vec x = arg;
    int wStart = 0;
    for (int i = 0; i < dims.length - 1; i++) {
      Mx w = new VecBasedMx(dims[i], weights.sub(wStart, dims[i] * dims[i + 1]));
      x = MxTools.multiply(w, x);
      wStart += dims[i] * dims[i + 1];
    }

    return x;
  }
}
