package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.models.nn.nodes.InputNode;

import static com.expleague.ml.models.nn.NeuralSpider.BackwardNode;
import static com.expleague.ml.models.nn.NeuralSpider.ForwardNode;

public interface InputLayerBuilder<InputType> extends LayerBuilder {
  void setInput(InputType input);
  int ydim(InputType input);
  InputLayer build();

  @Override
  default LayerBuilder setPrevBuilder(LayerBuilder layer) {
    return this;
  }

  @Override
  default LayerBuilder wStart(int wStart) {
    return this;
  }

  interface InputLayer extends Layer {
    void toState(Vec state);

    @Override
    default int xdim() {
      return 0;
    }

    @Override
    default int wdim() {
      return 0;
    }

    @Override
    default Seq<ForwardNode> forwardFlow() {
      return ArraySeq.iterate(ForwardNode.class, new InputNode(), ydim());
    }

    @Override
    default Seq<BackwardNode> backwardFlow() {
      return ArraySeq.emptySeq(BackwardNode.class);
    }

    @Override
    default Seq<BackwardNode> gradientFlow() {
      return ArraySeq.emptySeq(BackwardNode.class);
    }
  }
}
