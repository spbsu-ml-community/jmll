package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.models.nn.layers.*;
import org.junit.Test;

public class LeNetTest {
  private static final ConvNet leNet = createNet();

  private static ConvNet createNet() {
    ConvLayerBuilder conv1 = ConvLayerBuilder.create()
        .channels(20)
        .ksize(5, 5)
        .weightFill(FillerType.XAVIER);

    PoolLayerBuilder pool1 = PoolLayerBuilder.create()
        .ksize(2, 2).stride(2, 2);

    ConvLayerBuilder conv2 = ConvLayerBuilder.create()
        .channels(50)
        .ksize(5, 5)
        .weightFill(FillerType.XAVIER);

    PoolLayerBuilder pool2 = PoolLayerBuilder.create()
        .ksize(2, 2).stride(2, 2);

    FCLayerBuilder fc1 = FCLayerBuilder.create()
        .nOut(500)
        .weightFill(FillerType.XAVIER);

    FCLayerBuilder fc2 = FCLayerBuilder.create()
        .nOut(10)
        .weightFill(FillerType.XAVIER);


    NetworkBuilder<Vec>.Network network = new NetworkBuilder<>(
        new ConstSizeInputBuilder(28, 28))
        .addSeq(conv1, pool1, conv2, pool2, fc1, fc2)
        .build(new OneOutLayer(), fc2);

    return new ConvNet(network);
  }

  @Test
  public void inceptionTest() {
    ConvLayerBuilder conv5 = ConvLayerBuilder.create().ksize(5, 5).channels(5);
    ConvLayerBuilder conv3 = ConvLayerBuilder.create().ksize(3, 3).channels(5);
    ConvLayerBuilder conv1 = ConvLayerBuilder.create().ksize(1, 1).channels(5);
    MergeLayerBuilder merge = MergeLayerBuilder.create().layers(conv1, conv3, conv5);

    final NetworkBuilder<Vec> builder = new NetworkBuilder<>(new ConstSizeInputBuilder());
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
    Vec sample = new ArrayVec(28 * 28);
    VecTools.fill(sample, 1.);
    leNet.apply(sample);
  }
}
