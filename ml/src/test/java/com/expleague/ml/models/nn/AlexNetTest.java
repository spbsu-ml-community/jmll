package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.generic.ReLU;
import com.expleague.ml.models.nn.layers.*;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

public class AlexNetTest {
  private static final ConvNet alexNet = createNet();

  @NotNull
  private static ConvNet createNet() {
    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
        new ConstSizeInput3D(224, 224, 3))
        .addSeq(
            ConvLayerBuilder.create()
                .channels(64).ksize(11, 11).stride(4, 4).padd(2, 2).activation(ReLU.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            ConvLayerBuilder.create()
                .channels(192).ksize(5, 5).padd(2, 2).activation(ReLU.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            ConvLayerBuilder.create()
                .channels(384).ksize(3, 3).padd(2, 2).activation(ReLU.class),
            ConvLayerBuilder.create()
                .channels(256).ksize(3, 3).activation(ReLU.class),
            PoolLayerBuilder.create()
                .ksize(3, 3).stride(2, 2),
            FCLayerBuilder.create()
                .nOut(4096).activation(ReLU.class),
            FCLayerBuilder.create()
                .nOut(4096).activation(ReLU.class),
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
}
