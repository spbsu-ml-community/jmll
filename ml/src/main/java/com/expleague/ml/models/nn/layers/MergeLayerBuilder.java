package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.models.nn.NeuralSpider;
import com.expleague.ml.models.nn.NeuralSpider.ForwardNode;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MergeLayerBuilder implements LayerBuilder {
  private final List<LayerBuilder> prevLayers = new ArrayList<>();
  private int yStart;
  private int wStart;
  private MergeLayer layer;

  private MergeLayerBuilder() {}

  public static MergeLayerBuilder create() {
    return new MergeLayerBuilder();
  }

  @Override
  public LayerBuilder setPrevBuilder(LayerBuilder layer) {
    prevLayers.add(layer);
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

  @Override
  public MergeLayer getLayer() {
    return layer;
  }

  @Override
  public MergeLayer build() {
    if (layer != null) {
      return layer;
    }

    Layer[] layers = new Layer[prevLayers.size()];
    for (int i = 0; i < layers.length; i++) {
      layers[i] = prevLayers.get(i).getLayer();
      if (layers[i] == null) {
        throw new IllegalStateException("Graph is not acyclic");
      }
    }

    layer = new MergeLayer(layers);

    return layer;
  }

  public MergeLayerBuilder layers(LayerBuilder... layers) {
    Collections.addAll(prevLayers, layers);
    return this;
  }

  public class MergeLayer implements Layer {
    private final int xdim;
    private final int ydim;
    private final Layer[] prevLayers;

    private MergeLayer(Layer... prevLayers) {
      /*
       * TODO: check dimensionality if Layer3D
       */
      int sumX = 0;
      int sumY = 0;

      for (Layer layer: prevLayers) {
        sumX += layer.xdim();
        sumY += layer.ydim();
      }

      xdim = sumX;
      ydim = sumY;

      this.prevLayers = prevLayers;
    }

    @Override
    public int xdim() {
      return xdim;
    }

    @Override
    public int ydim() {
      return ydim;
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
    public void initWeights(Vec weights) { }

    @Override
    public Seq<ForwardNode> forwardFlow() {
      throw new NotImplementedException();
    }

    @Override
    public Seq<NeuralSpider.BackwardNode> backwardFlow() {
      return null;
    }

    @Override
    public Seq<NeuralSpider.BackwardNode> gradientFlow() {
      return null;
    }
  }
}
