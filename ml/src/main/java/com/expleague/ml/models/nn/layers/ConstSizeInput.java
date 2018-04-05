package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

import java.util.Arrays;

public class ConstSizeInput implements InputLayerBuilder<Vec> {
  private Vec input;
  private final int ydim;
  private int yStart;
  private final int[] dims;

  public ConstSizeInput(int... dims) {
    ydim = Arrays.stream(dims).reduce(1, (a, b) -> a * b);
    if (ydim <= 0) {
      throw new IllegalArgumentException("dims product must be greater than zero");
    }
    this.dims = dims;
  }

  private final InputLayer inputLayer = new InputLayer() {
    @Override
    public void toState(Vec state) {
      VecTools.assign(state.sub(yStart, ydim), input);
    }

    @Override
    public int xdim() {
      return ydim;
    }

    @Override
    public int ydim() {
      return ydim;
    }

    @Override
    public int yStart() {
      return yStart;
    }

    @Override
    public void initWeights(Vec weights) { }
  };

  @Override
  public void setInput(Vec input) {
    this.input = input;
  }

  @Override
  public int ydim(Vec input) {
    return input.dim();
  }

  @Override
  public Layer getLayer() {
    return inputLayer;
  }

  @Override
  public LayerBuilder yStart(int yStart) {
    this.yStart = yStart;
    return this;
  }

  @Override
  public InputLayer build() {
    return inputLayer;
  }

  @Override
  public String toString() {
    return "Input " + Arrays.toString(dims) + "\n";
  }
}
