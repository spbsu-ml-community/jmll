package com.expleague.text;

import com.expleague.commons.func.Functions;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.data.tools.FeatureSet;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.FeatureMetaImpl;

import java.util.Objects;

public class EmbeddingFeatureSet extends FeatureSet.Stub<TextItem> {
  private final Embedding<CharSeq> embedding;
  private CharSeq nextText;

  public EmbeddingFeatureSet(String prefix, String description, Embedding<CharSeq> embedding) {
    super(generateMeta(prefix, description, embedding.dim()));
    this.embedding = embedding;
  }

  @Override
  public void accept(TextItem item) {
    super.accept(item);
    nextText = item.text();
  }

  @Override
  public Vec advanceTo(Vec to) {
    VecTools.scale(to, 0);
    final int docLen = CharSeqTools.split(nextText, " ", false)
        .map(Functions.cast(CharSeq.class)).map(embedding)
        .filter(Objects::nonNull).peek(v -> VecTools.append(to, v)).mapToInt(x -> 1).sum();
    VecTools.scale(to, 1. / (0.5 + docLen));
    return to;
  }

  private static FeatureMetaImpl[] generateMeta(String prefix, String description, int dim) {
    final FeatureMetaImpl[] result = new FeatureMetaImpl[dim];
    for (int i = 0; i < dim; i++) {
      result[i] = new FeatureMetaImpl(prefix + "-" + i, description + "#" + i, FeatureMeta.ValueType.VEC);
    }
    return result;
  }
}
