package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;

public interface OutputLayerBuilder extends LayerBuilder {
  OutputLayer getLayer();
  OutputLayer build();

  @Override
  default LayerBuilder yStart(int yStart) {
    return this;
  }

  @Override
  default LayerBuilder wStart(int wStart) {
    return this;
  }


  interface OutputLayer extends Layer {
    Vec fromState(Vec state);

    @Override
    default int ydim() {
      return xdim();
    }

    @Override
    default int wdim() {
      return 0;
    }

    @Override
    default int yStart() {
      return -1;
    }

    @Override
    default void initWeights(Vec weights) { }
  }
}
