//package com.expleague.ml.models.nn.nodes;
//
//import com.expleague.commons.math.vectors.Vec;
//import com.expleague.commons.math.vectors.VecTools;
//import com.expleague.ml.models.nn.NeuralSpider;
//import com.expleague.ml.models.nn.layers.ConvLayer;
//
//public class Conv2DNode implements NeuralSpider.Node {
//  private final ConvLayer owner;
//  private final int channelSize;
//  private final int width;
//  private final int kSizeX;
//  private final int kSizeY;
//
//  public Conv2DNode(ConvLayer owner) {
//    this.owner = owner;
//    channelSize = owner.getChannelSize();
//    width = owner.getChannelWidth();
//    kSizeX = owner.getKSizeX();
//    kSizeY = owner.getKSizeY();
//  }
//
//  /**
//   * It is assumed that data is stored
//   * in (Channels, Height, Weight) view
//   */
//  @Override
//  public double apply(Vec state, Vec betta) {
//    double result = 0.;
//    for (int c = 0; c < owner.getNumInputChannels(); c++) {
//      for (int i = 0; i < kSizeX; i++) {
//        for (int j = 0; j < kSizeY; j++) {
//          final int idx = i * owner.getStrideX() * width + j * owner.getStrideY();
//          result += state.get(c * channelSize + idx)
//              * betta.get(c * kSizeX * kSizeY + i * kSizeY + j);
//        }
//      }
//    }
//
//    return result;
//  }
//
//  @Override
//  public void gradByStateTo(Vec state, Vec betta, Vec to) {
//    VecTools.assign(to, betta);
//  }
//
//  @Override
//  public void gradByParametersTo(Vec state, Vec betta, Vec to) {
//    // TODO!
////    for (int i = 0; i < ...; i++) {
////      to.adjust()
////    }
//  }
//}
