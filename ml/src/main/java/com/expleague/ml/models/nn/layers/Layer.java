package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;

import java.util.stream.Stream;

public interface Layer {
  int xdim();
  int ydim();
  int wdim();
  int yStart();
  void initWeights(Vec weights);
  Seq<NodeCalcer> materialize();
}
