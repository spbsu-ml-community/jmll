package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.ArraySeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqBuilder;
import com.expleague.ml.models.nn.nodes.PoolNode;

import java.util.stream.IntStream;

import static com.expleague.ml.models.nn.NeuralSpider.*;

public class PoolLayerBuilder extends ConvLayerBuilder {
  private int kSizeX = 3;
  private int kSizeY = 3;
  private int strideX = 3;
  private int strideY = 3;
  private PoolLayer layer;
  private int yStart;
  private LayerBuilder prevBuilder;

  private PoolLayerBuilder() {
    super();
  }

  public static PoolLayerBuilder create() {
    return new PoolLayerBuilder();
  }

  public PoolLayerBuilder ksize(int kSizeX, int kSizeY) {
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;
    return this;
  }

  public PoolLayerBuilder stride(int strideX, int strideY) {
    this.strideX = strideX;
    this.strideY = strideY;
    return this;
  }

  @Override
  public PoolLayer getLayer() {
    return layer;
  }

  @Override
  public LayerBuilder setPrevBuilder(LayerBuilder prevBuilder) {
    this.prevBuilder = prevBuilder;
    return this;
  }

  @Override
  public LayerBuilder yStart(int yStart) {
    this.yStart = yStart;
    return this;
  }

  @Override
  public LayerBuilder wStart(int wStart) {
    return this;
  }

  @Override
  public PoolLayer build() {
    if (prevBuilder.getLayer() == null) {
      throw new IllegalStateException("Graph is not acyclic");
    }

    if (layer != null) {
      return layer;
    }

    layer = new PoolLayer((Layer3D) prevBuilder.getLayer());
    return layer;
  }

  public class PoolLayer extends ConvLayer {
    private final PoolNode node;

    private PoolLayer(Layer3D in) {
      super(in);
       node = new PoolNode(yStart, input.yStart(), input.channels(),
          input.width(), width(), channels(), kSizeX, kSizeY, strideX, strideY);
    }

    @Override
    public int wdim() {
      return 0;
    }

    @Override
    public void initWeights(Vec weights) { }

    @Override
    public Seq<ForwardNode> forwardFlow() {
      return ArraySeq.iterate(ForwardNode.class, node.forward(), ydim());
    }

    @Override
    public Seq<BackwardNode> backwardFlow() {
      return ArraySeq.iterate(BackwardNode.class, node.backward(), xdim());
    }

    @Override
    public Seq<BackwardNode> gradientFlow() {
      return  ArraySeq.iterate(BackwardNode.class, node.gradient(), wdim());
    }
  }
}
