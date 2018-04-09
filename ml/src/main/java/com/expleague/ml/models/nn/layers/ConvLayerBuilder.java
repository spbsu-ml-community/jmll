package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqBuilder;
import com.expleague.ml.models.nn.Identity;
import com.expleague.ml.models.nn.nodes.ConvCalcer;

import java.util.stream.IntStream;

import static com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;

public class ConvLayerBuilder implements LayerBuilder {
  private int kSizeX = 3;
  private int kSizeY = 3;
  private int strideX = 1;
  private int strideY = 1;
  private int paddX = 0;
  private int paddY = 0;
  private int outChannels = 0;
  private FillerType fillerType = FillerType.NORMAL;
  private LayerBuilder prevBuilder;
  private ConvLayer layer;
  private int yStart;
  private int wStart;
  private AnalyticFunc activation = new Identity();

  protected ConvLayerBuilder() {}

  public static ConvLayerBuilder create() {
    return new ConvLayerBuilder();
  }

  public ConvLayerBuilder ksize(int kSizeX, int kSizeY) {
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;

    if (paddX != 0 || paddY != 0) {
      paddX = (kSizeX - 1) / 2;
      paddY = (kSizeY - 1) / 2;
    }

    return this;
  }

  public ConvLayerBuilder stride(int strideX, int strideY) {
    this.strideX = strideX;
    this.strideY = strideY;
    return this;
  }

  public ConvLayerBuilder channels(int channels) {
    this.outChannels = channels;
    return this;
  }

  public ConvLayerBuilder samePadd() {
    paddX = (kSizeX - 1) / 2;
    paddY = (kSizeY - 1) / 2;
    return this;
  }

  public ConvLayerBuilder padd(int paddX, int paddY) {
    this.paddX = paddX;
    this.paddY = paddY;
    return this;
  }

  @Override
  public ConvLayer getLayer() {
    return layer;
  }

  @Override
  public LayerBuilder setPrevBuilder(LayerBuilder prevBuilder) {
    if (this.prevBuilder != null) {
      throw new IllegalStateException("Conv layer can have only one previous layer");
    }

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
    this.wStart = wStart;
    return this;
  }

  public ConvLayerBuilder weightFill(FillerType fillerType) {
    this.fillerType = fillerType;
    return this;
  }

  @Override
  public Layer3D build() {
    if (prevBuilder.getLayer() == null) {
      throw new IllegalStateException("Graph is not acyclic");
    }

    if (layer != null) {
      return layer;
    }

    if (outChannels == 0) {
      throw new IllegalStateException("The number of output channels is not provided");
    }

    layer = new ConvLayer((Layer3D) prevBuilder.getLayer());

    return layer;
  }

  public ConvLayerBuilder activation(Class<? extends AnalyticFunc> actClass) {
    try {
      activation = actClass.newInstance();
    }
    catch (InstantiationException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
    return this;
  }

  public class ConvLayer implements Layer3D {
    protected final Layer3D input;
    private final Filler filler;

    protected ConvLayer(Layer3D input) {
      this.input = input;
      filler = FillerType.getInstance(fillerType, this);
    }

    public void initWeights(Vec weights) {
      filler.apply(weights.sub(wStart, wdim()));
    }

    @Override
    public int height() {
      return (input.height() + 2 * paddX - kSizeX) / strideX + 1;
    }

    @Override
    public int width() {
      return (input.width() + 2 * paddY - kSizeY) / strideY + 1;
    }

    @Override
    public int channels() {
      return outChannels;
    }

    @Override
    public int xdim() {
      return input.channels() * input.width() * input.height();
    }

    @Override
    public int ydim() {
      return width() * height() * outChannels;
    }

    @Override
    public int wdim() {
      return input.channels() * kSizeX * kSizeY * outChannels;
    }

    @Override
    public int yStart() {
      return yStart;
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

    @Override
    public Seq<NodeCalcer> materialize() {
      final NodeCalcer calcer = new ConvCalcer(
          yStart, wStart, input.yStart(), input.width(), width(),
          kSizeX, kSizeY, strideX, strideY, paddX, paddY,
          input.channels(), outChannels, activation);
      final SeqBuilder<NodeCalcer> seqBuilder = new ArraySeqBuilder<>(NodeCalcer.class);
      IntStream.range(0, ydim()).forEach(i -> seqBuilder.add(calcer));
      return seqBuilder.build();
    }

    @Override
    public String toString() {
      return "Conv outSize[" + height() + ", " + width() + ", " + channels() + "] " +
          "kernel[" + kSizeX + ", " + kSizeY + "] " +
          "stride[" + strideX + ", " + strideY + "]\n";
    }
  }

}
