package com.expleague.ml.embedding;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.Metric;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.glove.GloVeBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import org.jetbrains.annotations.Nullable;

import java.io.BufferedWriter;
import java.io.Writer;
import java.nio.file.Path;
import java.util.function.Function;

public interface Embedding<T> extends Function<T, Vec>, Metric<T> {
  @Nullable
  Vec apply(T to);

  interface Builder<T> {
    Builder<T> file(Path path);
    Builder<T> minWordCount(int count);
    Builder<T> leftWindow(int wnd);
    Builder<T> rightWindow(int wnd);

    Embedding<T> build();
  }

  enum Type {
    GLOVE(GloVeBuilder.class),
    DECOMP(DecompBuilder.class)
    ;

    private final Class<? extends Builder> builderClass;

    Type(Class<? extends Builder> builderClass) {
      this.builderClass = builderClass;
    }
  }

  static void write(Embedding what, Writer to) {
    ((EmbeddingImpl) what).write(to);
  }

  static Builder builder(Type type) {
    try {
      return type.builderClass.newInstance();
    }
    catch (InstantiationException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }
}
