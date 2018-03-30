package com.expleague.ml.models.nn.layers;

public interface LayerBuilder {
  Layer getLayer();
  LayerBuilder setPrevBuilder(LayerBuilder layer);
  LayerBuilder yStart(int yStart);
  LayerBuilder wStart(int wStart);

  Layer build();
}
