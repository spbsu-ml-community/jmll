package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.models.nn.nodes.PoolNode;

import static com.expleague.ml.models.nn.NeuralSpider.BackwardNode;
import static com.expleague.ml.models.nn.NeuralSpider.ForwardNode;

public class PoolLayerBuilder implements LayerBuilder {
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
    assert(kSizeX > 0);
    assert(kSizeY > 0);
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;
    return this;
  }

  public PoolLayerBuilder stride(int strideX, int strideY) {
    assert(strideX > 0);
    assert(strideY > 0);
    this.strideX = strideX;
    this.strideY = strideY;
    return this;
  }

  @Override
  public Layer3D getLayer() {
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

  public class PoolLayer implements Layer3D {
    private final PoolNode node;
    private final Layer3D input;

    private PoolLayer(Layer3D in) {
      input = in;
      node = new PoolNode(yStart, input.yStart(), input.channels(),
          input.width(), width(), height(), kSizeX, kSizeY, strideX, strideY);
    }

    @Override
    public int wdim() {
      return 0;
    }

    @Override
    public int yStart() {
      return yStart;
    }

    @Override
    public int xdim() {
      return input.ydim();
    }

    @Override
    public int ydim() {
      return width() * height() * channels();
    }

    @Override
    public int height() {
      return (input.height() - kSizeX) / strideX + 1;
    }

    @Override
    public int width() {
      return (input.width() - kSizeY) / strideY + 1;
    }

    @Override
    public int channels() {
      return input.channels();
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

    @Override
    public String toString() {
      return "Pool outSize[" + height() + ", " + width() + ", " + channels() + "] " +
          "kernel[" + kSizeX + ", " + kSizeY + "] stride[" + strideX + ", " + strideY + "]\n";
    }

    public int kSizeX() {
      return kSizeX;
    }

    public int kSizeY() {
      return kSizeY;
    }

    public int strideX() {
      return strideX;
    }

    public int strideY() {
      return strideY;
    }
  }
}
