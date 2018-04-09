package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqBuilder;
import com.expleague.ml.models.nn.Identity;
import com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;
import com.expleague.ml.models.nn.nodes.FCCalcer;

import java.util.stream.IntStream;

public class FCLayerBuilder implements LayerBuilder {
  private int nOut;
  private FillerType fillerType = FillerType.CONSTANT;
  private LayerBuilder prevBuilder;
  private int yStart;
  private int wStart;
  private FCLayer layer;
  private AnalyticFunc activation = new Identity();

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

  public FCLayerBuilder activation(Class<? extends AnalyticFunc> actClass) {
    try {
      activation = actClass.newInstance();
    }
    catch (InstantiationException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
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
    private final Filler filler;

    private FCLayer(Layer input) {
      this.input = input;
      filler = FillerType.getInstance(fillerType, this);
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
      final NodeCalcer calcer = new FCCalcer(
          yStart, ydim(), input.yStart(), xdim(), wStart, wdim(), activation);
      final SeqBuilder<NodeCalcer> seqBuilder = new ArraySeqBuilder<>(NodeCalcer.class);
      IntStream.range(0, ydim()).forEach(i -> seqBuilder.add(calcer));
      return seqBuilder.build();
    }

    @Override
    public String toString() {
      return "FC [" + input.ydim() + ", " + nOut + "]\n";
    }
  }
}
