package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.generic.Sigmoid;
import com.expleague.ml.func.generic.Sum;
import com.expleague.ml.models.nn.layers.ConstSizeInput3D;
import com.expleague.ml.models.nn.layers.ConvLayerBuilder;
import com.expleague.ml.models.nn.layers.OneOutLayer;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class ConvGradTest {
  private static final NeuralSpider<Vec> spider = new NeuralSpider<>();
  private static final FastRandom rng = new FastRandom();
  private static final int ROUNDS = 10;
  private static final double EPS = 1e-6;
  private static final double SCALE = 1e-2;

  @Test
  public void convLayerTest() {
    final int width = 15;
    final int height = 15;
    final int channels = 5;

    for (int ksizeX = 1; ksizeX <= height; ksizeX++) {
      for (int ksizeY = 1; ksizeY <= width; ksizeY++) {
        for (int strideX = 1; strideX < height - ksizeX; strideX++) {
          for (int strideY = 1; strideY < width - ksizeY; strideY++) {

            System.out.println("kSize [" + ksizeX + ", " + ksizeY + "], " +
                "stride [" + strideX + ", " + strideY + "]");
            final NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
                new ConstSizeInput3D(height, width, channels))
                .append(ConvLayerBuilder.create()
                    .ksize(ksizeX, ksizeY)
                    .stride(strideX, strideY)
                    .channels(rng.nextInt(5) + 1))
                .build(new OneOutLayer());

            testNetwork(network, width, height, channels);
          }
        }
      }
    }
  }

  @Test
  public void multiLayerTest() {
    final int width = 100;
    final int height = 100;
    final int channels = 3;

    for (int shot = 0; shot < 50; shot++) {
      final int numLayers = rng.nextInt(4) + 2;

      NetworkBuilder<Vec> builder = new NetworkBuilder<>(new ConstSizeInput3D(height, width, channels));

      int curWidth = width;
      int curHeight = height;
      int prevChannels = channels;
      for (int i = 0; i < numLayers; i++) {
        final int ksizeX = rng.nextPoisson(3.) + 1;
        final int ksizeY = rng.nextPoisson(3.) + 1;
        final int strideX = rng.nextInt(2) + 1;
        final int strideY = rng.nextInt(2) + 1;

        if (ksizeX >= curHeight || ksizeY >= curWidth)
          continue;

        curWidth = (curWidth - ksizeY) / strideY + 1;
        curHeight = (curHeight - ksizeX) / strideX + 1;
        if (curHeight <= 0 || curWidth <= 0)
          continue;
        prevChannels = rng.nextPoisson(1.) + prevChannels;

        ConvLayerBuilder convLayerBuilder = ConvLayerBuilder.create()
            .ksize(ksizeX, ksizeY).stride(strideX, strideY)
            .channels(prevChannels);
        if (i != numLayers - 1) {
          convLayerBuilder.activation(Sigmoid.class);
        }
        builder.append(convLayerBuilder);
      }

      final NetworkBuilder<Vec>.Network network = builder.build(new OneOutLayer());
      System.out.println(network);
      testNetwork(network, width, height, channels);
    }
  }

  private void testNetwork(NetworkBuilder<Vec>.Network network, int... dims) {
    final Vec weights = new ArrayVec(network.wdim());
    final Vec weightsCopy = new ArrayVec(network.wdim());
    Vec arg = new ArrayVec(dims[0] * dims[1] * dims[2]);
    Vec gradWeight = new ArrayVec(network.wdim());

    for (int i = 0; i < 5; i++) {
      VecTools.fillUniform(weights, rng, SCALE);

      VecTools.assign(weightsCopy, weights);

      VecTools.fillUniform(arg, rng, SCALE);

      final Vec state = spider.compute(network, arg, weights);
      final double stateSum = VecTools.sum(state);

      spider.parametersGradient(network, arg, new Sum(), weights, gradWeight);

      for (int round = 0; round < ROUNDS; round++) {
        final int wIdx = rng.nextInt(network.wdim());
        weightsCopy.adjust(wIdx, EPS);
        final double incState = VecTools.sum(spider.compute(network, arg, weightsCopy));
        double grad = (incState - stateSum) / EPS;

        assertEquals("Test idx " + wIdx, grad, gradWeight.get(wIdx), EPS * 100);

        weightsCopy.adjust(wIdx, -EPS);
      }
    }
  }
}
