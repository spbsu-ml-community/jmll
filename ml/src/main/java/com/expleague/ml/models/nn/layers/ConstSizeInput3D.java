package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

public class ConstSizeInput3D implements InputLayerBuilder<Vec> {
  private int yStart;
  private Vec input;
  private final int ydim;
  private final InputLayer layer = new Input3D();
  private final int height;
  private final int width;
  private final int channels;

  @Override
  public Layer getLayer() {
    return layer;
  }

  @Override
  public LayerBuilder yStart(int yStart) {
    this.yStart = yStart;
    return this;
  }

  private int getyStart() {
    return yStart;
  }

  public ConstSizeInput3D(int height, int width, int channels) {
    ydim = height * width * channels;

    this.height = height;
    this.width = width;
    this.channels = channels;

    if (ydim <= 0) {
      throw new IllegalArgumentException("dims product must be greater than zero");
    }
  }

  @Override
  public void setInput(Vec input) {
    this.input = input;
  }

  @Override
  public int ydim(Vec input) {
    return ydim;
  }

  @Override
  public InputLayer build() {
    return layer;
  }

  public class Input3D implements InputLayer, Layer3D {
    @Override
    public void toState(Vec state) {
      VecTools.assign(state.sub(yStart, input.dim()), input);
    }

    @Override
    public int height() {
      return height;
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public int channels() {
      return channels;
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
      return getyStart();
    }

    @Override
    public void initWeights(Vec weights) { }

    @Override
    public String toString() {
      return "Input [" + height + ", " + width + ", " + channels + "]\n";
    }
  }
}
