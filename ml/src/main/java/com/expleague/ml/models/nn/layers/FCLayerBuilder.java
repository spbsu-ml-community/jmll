package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqBuilder;
import com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;
import com.expleague.ml.models.nn.nodes.FCCalcer;

import java.util.stream.IntStream;

public class FCLayerBuilder implements LayerBuilder {
  private int nOut = 0;
  private FillerType fillerType;
  private LayerBuilder prevBuilder;
  private int yStart;
  private int wStart;
  private FCLayer layer;

  public static FCLayerBuilder create() {
    return new FCLayerBuilder();
  }

  public FCLayerBuilder nOut(int nOut) {
    this.nOut = nOut;
    return this;
  }

  public FCLayerBuilder weightFill(FillerType fillerType) {
    this.fillerType = fillerType;
    return this;
  }

  @Override
  public LayerBuilder setPrevBuilder(LayerBuilder prevBuilder) {
    if (this.prevBuilder != null) {
      throw new IllegalStateException("FCLayer can have only one prev layer");
    }

    this.prevBuilder = prevBuilder;
    return this;
  }

  @Override
  public FCLayer getLayer() {
    return layer;
  }

  @Override
  public FCLayerBuilder yStart(int yStart) {
    this.yStart = yStart;
    return this;
  }

  @Override
  public FCLayerBuilder wStart(int wStart) {
    this.wStart = wStart;
    return this;
  }

  @Override
  public Layer build() {
    if (prevBuilder.getLayer() == null) {
      throw new IllegalStateException("Graph is not acyclic");
    }

    if (layer != null) {
      return layer;
    }

    layer = new FCLayer(prevBuilder.getLayer());
    return layer;
  }

  public class FCLayer implements Layer {
    private final Layer input;
    private final Filler filler = FillerType.getInstance(fillerType, this);

    private FCLayer(Layer input) {
      this.input = input;
    }

    public void initWeights(Vec weights) {
      filler.apply(weights.sub(wStart, wdim()));
    }

    @Override
    public int xdim() {
      return input.ydim();
    }

    @Override
    public int ydim() {
      return nOut;
    }

    @Override
    public int wdim() {
      return xdim() * ydim();
    }

    @Override
    public int yStart() {
      return yStart;
    }

    @Override
    public Seq<NodeCalcer> materialize() {
      final NodeCalcer calcer = new FCCalcer(yStart, ydim(), input.yStart(), xdim(), wStart, wdim());
      final SeqBuilder<NodeCalcer> seqBuilder = new ArraySeqBuilder<>(NodeCalcer.class);
      IntStream.range(0, ydim()).forEach(i -> seqBuilder.add(calcer));
      return seqBuilder.build();
    }
  }
}
