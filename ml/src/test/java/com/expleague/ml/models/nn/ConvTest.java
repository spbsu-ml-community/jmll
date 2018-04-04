package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.models.nn.layers.ConstSizeInput3D;
import com.expleague.ml.models.nn.layers.ConvLayerBuilder;
import com.expleague.ml.models.nn.layers.OneOutLayer;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;

public class ConvTest {
  private static final NeuralSpider<Vec> spider = new NeuralSpider<>();

  @Test
  public void convLayerTest() {
    for (int ksizeX = 1; ksizeX < 7; ksizeX += 2) {
      for (int ksizeY = 1; ksizeY < 7; ksizeY += 2) {
        NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
            new ConstSizeInput3D(10, 10, 1))
            .append(ConvLayerBuilder.create().ksize(ksizeX, ksizeY).channels(1))
            .build(new OneOutLayer());

        testNetwork(network, 10, 10, ksizeX, ksizeY);
      }
    }
  }

  public void testNetwork(NetworkBuilder<Vec>.Network network, int width, int height,
                          int ksizeX, int ksizeY) {
    Vec weights = new ArrayVec(ksizeX * ksizeY);
    VecTools.fill(weights, 1.);
//    for (int i = 0; i < weights.dim(); i++) {
//      weights.set(i, i);
//    }

    Vec arg = new ArrayVec(width * height);
    VecTools.fill(arg, 1.);

    final int dstWidth = height - ksizeX + 1;
    final int dstHeight = width - ksizeY + 1;
    Vec expect = new ArrayVec(dstHeight * dstWidth);
    VecTools.fill(expect, ksizeX * ksizeY);
    final Vec result = spider.compute(network, arg, weights);
//    final Vec result = conv(arg, weights, ksizeX, ksizeY, width, height);

    assertTrue(VecTools.equals(expect, result));
  }

  private Vec conv(Vec arg, Vec weights, int ksizeX, int ksizeY, int width, int height) {
    final int cX = ksizeX / 2;
    final int cY = ksizeY / 2;

//    (height + 2 * paddX - ksizeX) / strideX + 1
    final int dstHeight = height - ksizeX + 1;
    final int dstWidth = width - ksizeY + 1;
    Vec result = new ArrayVec(dstHeight * dstWidth);

    for (int x = cX; x < height - cX; x++) {
      for (int y = cY; y < width - cY; y++) {
        double c = conv(x, y, arg, weights, cX, cY, width, height);
        result.set((x - cX) * dstWidth + (y - cY), c);
      }
    }

    return result;
  }

  private double conv(int x, int y, Vec arg, Vec weights, int cX, int cY, int width, int height) {
    double result = 0.;
    for (int i = -cX; i <= cX; i++) {
      for (int j = -cY; j <= cY; j++) {
        result += arg.get((x + i) * width + y + j)
            * weights.get((i + cX) * (2 * cY + 1) + j + cY);
      }
    }

    return result;
  }
}
