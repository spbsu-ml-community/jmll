package com.expleague.ml.models.nn;

import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.generic.Sigmoid;
import com.expleague.ml.func.generic.Sum;
import com.expleague.ml.models.nn.layers.*;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class AlexNetTest {
  private static final ConvNet alexNet = createNet();

  @NotNull
  private static ConvNet createNet() {
    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
        new ConstSizeInput3D(224, 224, 3))
        .addSeq(
            ConvLayerBuilder.create()
                .channels(64).ksize(11, 11).stride(4, 4).padd(2, 2).activation(Sigmoid.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            ConvLayerBuilder.create()
                .channels(192).ksize(5, 5).padd(2, 2).activation(Sigmoid.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            ConvLayerBuilder.create()
                .channels(384).ksize(3, 3).padd(2, 2).activation(Sigmoid.class),
            ConvLayerBuilder.create()
                .channels(256).ksize(3, 3).activation(Sigmoid.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            FCLayerBuilder.create()
                .nOut(4096).activation(Sigmoid.class),
            FCLayerBuilder.create()
                .nOut(4096).activation(Sigmoid.class),
            FCLayerBuilder.create()
                .nOut(1000))
        .build(new OneOutLayer());

    return new ConvNet(network);
  }

  @Test
  public void inceptionTest() {
    ConvLayerBuilder conv5 = ConvLayerBuilder.create().ksize(5, 5).channels(5);
    ConvLayerBuilder conv3 = ConvLayerBuilder.create().ksize(3, 3).channels(5);
    ConvLayerBuilder conv1 = ConvLayerBuilder.create().ksize(1, 1).channels(5);
    MergeLayerBuilder merge = MergeLayerBuilder.create().layers(conv1, conv3, conv5);

    final NetworkBuilder<Vec> builder = new NetworkBuilder<>(new ConstSizeInput());
    final InputLayerBuilder<Vec> input = builder.input();

    builder.connect(input, conv1)
        .connect(input, conv3)
        .connect(input, conv5)
        .append(merge)
        .append(FCLayerBuilder.create().nOut(10))
        .build(new OneOutLayer(), builder.last());
  }

  @Test
  public void forwardTest() {
    Vec sample = new ArrayVec(224 * 224 * 3);
    VecTools.fill(sample, 1.);
    alexNet.apply(sample);
  }

  private static final double EPS = 1e-6;
  private static final FastRandom rng = new FastRandom();

  @Test
  public void backwardTest() {
    final TransC1 target = new Sum();
    alexNet.setTarget(target);
    final int wdim = alexNet.wdim();
    final Vec weights = new ArrayVec(wdim);
    final Vec weightsCopy = new ArrayVec(wdim);
    final Vec arg = new ArrayVec(224 * 224 * 3);

    for (int i = 0; i < 5; i++) {
      System.out.println("test no " + i);
      VecTools.fillUniform(weights, rng);

      VecTools.assign(weightsCopy, weights);

      VecTools.fillUniform(arg, rng);

      alexNet.setWeights(weights);
      final Vec state = alexNet.apply(arg);
      final double stateSum = VecTools.sum(state);

      Vec gradWeight = alexNet.gradient(arg);

      for (int round = 0; round < 10; round++) {
        System.out.println(round);
        final int wIdx = rng.nextInt(wdim);
        weightsCopy.adjust(wIdx, EPS);

        alexNet.setWeights(weightsCopy);
        final double incState = VecTools.sum(alexNet.apply(arg));
        double grad = (incState - stateSum) / EPS;

        assertEquals("Test idx " + wIdx, grad, gradWeight.get(wIdx), 1e-2);

        weightsCopy.adjust(wIdx, -EPS);
      }
    }
  }
}
