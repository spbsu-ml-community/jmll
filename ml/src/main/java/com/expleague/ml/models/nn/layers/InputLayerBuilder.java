package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.models.nn.NeuralSpider;
import com.expleague.ml.models.nn.nodes.InputNodeCalcer;

import java.util.stream.IntStream;

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
    default Seq<NeuralSpider.NodeCalcer> materialize() {
      final NeuralSpider.NodeCalcer calcer = new InputNodeCalcer();
      final ArraySeqBuilder<NeuralSpider.NodeCalcer> seqBuilder = new ArraySeqBuilder<>(NeuralSpider.NodeCalcer.class);
      IntStream.range(0, ydim()).forEach(i -> seqBuilder.add(calcer));
      return seqBuilder.build();
    }
  }
}
