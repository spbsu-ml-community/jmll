package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.ml.data.tools.impl.JoinedFeatureSet;
import com.expleague.ml.meta.FeatureMeta;

import java.util.ArrayList;
import java.util.List;

public interface FeatureSet {
  Vec advance();
  Vec advance(FeatureMeta... features);
  int dim();
  FeatureMeta meta(int index);

  static FeatureSet join(FeatureSet... fs) {
    return new JoinedFeatureSet(fs);
  }

  class Accumulator {
    private final FeatureSet fs;
    private final List<VecBuilder> builders;

    public Accumulator(FeatureSet fs) {
      this.fs = fs;
      this.builders = new ArrayList<>(fs.dim());
      for (int i = 0; i < fs.dim(); i++) {
        this.builders.add(new VecBuilder());
      }
    }

    public void advance() {
      final Vec vec = fs.advance();
      for (int i = 0; i < fs.dim(); i++) {
        builders.get(i).append(vec.get(i));
      }
    }

    public void splashOut(PoolBuilder builder) {
      for (int i = 0; i < fs.dim(); i++) {
        builder.newFeature(fs.meta(i), builders.get(i).build());
      }
    }
  }
}
