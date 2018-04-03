package com.expleague.ml.models.nn.layers;

public enum FillerType {
  NORMAL,
  XAVIER,
  CONSTANT;

  public static Filler getInstance(FillerType type, Layer layer) {
    switch (type) {
      case NORMAL: return new NormalFiller();
      case XAVIER: return new XavierFiller(layer.xdim(), layer.ydim());
      default: return new ConstFiller();
    }
  }
}
