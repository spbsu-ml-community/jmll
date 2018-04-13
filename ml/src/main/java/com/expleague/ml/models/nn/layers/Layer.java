package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.models.nn.NeuralSpider.BackwardNode;
import com.expleague.ml.models.nn.NeuralSpider.ForwardNode;

public interface Layer {
  interface Node {
    ForwardNode forward();
    BackwardNode backward();
    BackwardNode gradient();
  }

  int xdim();
  int ydim();
  int wdim();
  int yStart();
  void initWeights(Vec weights);
  Seq<ForwardNode> forwardFlow();
  Seq<BackwardNode> backwardFlow();
  Seq<BackwardNode> gradientFlow();
}
