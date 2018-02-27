//package com.expleague.ml.models.nn.nodes;
//
//import com.expleague.commons.math.vectors.Vec;
//import com.expleague.commons.math.vectors.VecTools;
//import com.expleague.ml.models.nn.NeuralSpider;
//
//public class LinearNode implements NeuralSpider.Node {
//  @Override
//  public double apply(Vec state, Vec betta) {
//    return VecTools.multiply(state, betta);
//  }
//
//  @Override
//  public void gradByStateTo(Vec state, Vec betta, Vec to) {
//    VecTools.assign(to, betta);
//  }
//
//  @Override
//  public void gradByParametersTo(Vec state, Vec betta, Vec to) {
//    VecTools.assign(to, state);
//  }
//}
