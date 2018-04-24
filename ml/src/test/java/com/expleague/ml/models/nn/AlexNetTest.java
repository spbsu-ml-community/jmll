package com.expleague.ml.models.nn;

import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.generic.ReLU;
import com.expleague.ml.func.generic.Sigmoid;
import com.expleague.ml.func.generic.Sum;
import com.expleague.ml.models.nn.layers.*;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class AlexNetTest {
  private static final ConvNet alexNet = createNet();
  private static final ConvNet leNet = createLeNet();

  private static ConvNet createLeNet() {
    NetworkBuilder<Vec>.Network network =
        new NetworkBuilder<>(new ConstSizeInput3D(28, 28, 1))
            .append(ConvLayerBuilder.create()
                .channels(20)
                .ksize(5,5)
                .weightFill(FillerType.XAVIER))
            .append(PoolLayerBuilder.create().ksize(2, 2).stride(2, 2))
            .append(ConvLayerBuilder.create()
                .channels(50)
                .ksize(5, 5)
                .weightFill(FillerType.XAVIER))
            .append(PoolLayerBuilder.create().ksize(2, 2).stride(2, 2))
            .append(FCLayerBuilder.create().nOut(500).weightFill(FillerType.XAVIER).activation(ReLU.class))
            .append(FCLayerBuilder.create().nOut(10).weightFill(FillerType.XAVIER))
            .build(new OneOutLayer());

    System.out.println(network);

    return new ConvNet(network);
  }

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

  @Test
  public void forwardLeNetTest() {
    Vec sample = new ArrayVec(28 * 28);
    FastRandom rng = new FastRandom();
    for (int i = 0; i < 30; i++) {
      VecTools.fillUniform(sample, rng);
      leNet.apply(sample);
    }
  }

  @Test
  public void backwardLeNetTest() {
    backwardTest(leNet, 1, 28, 28);
  }

  @Test
  public void backwardAlexNetTest() {
    backwardTest(alexNet, 3, 224, 224);
  }

  private static final double EPS = 1e-6;
  private static final FastRandom rng = new FastRandom();

  public void backwardTest(ConvNet nn, int channels, int width, int height) {
    final TransC1 target = new Sum();
    nn.setTarget(target);
    final int wdim = nn.wdim();
    final Vec weights = new ArrayVec(wdim);
    final Vec weightsCopy = new ArrayVec(wdim);
    final Vec arg = new ArrayVec(channels * width * height);

    for (int i = 0; i < 5; i++) {
      System.out.println("test no " + i);
      VecTools.fillUniform(weights, rng);

      VecTools.assign(weightsCopy, weights);

      VecTools.fillUniform(arg, rng);

      nn.setWeights(weights);
      final Vec state = nn.apply(arg);
      final double stateSum = VecTools.sum(state);

      Vec gradWeight = nn.gradient(arg);

      for (int round = 0; round < 10; round++) {
        System.out.println(round);
        final int wIdx = rng.nextInt(wdim);
        weightsCopy.adjust(wIdx, EPS);

        nn.setWeights(weightsCopy);
        final double incState = VecTools.sum(nn.apply(arg));
        double grad = (incState - stateSum) / EPS;

        assertEquals("Test idx " + wIdx, grad, gradWeight.get(wIdx), EPS * 100);

        weightsCopy.adjust(wIdx, -EPS);
      }
    }
  }
}
