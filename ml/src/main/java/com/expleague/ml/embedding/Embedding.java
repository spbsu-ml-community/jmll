package com.expleague.ml.embedding;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.Metric;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.decomp.MultiDecompBuilder;
import com.expleague.ml.embedding.glove.GloVeBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import org.jetbrains.annotations.Nullable;
import sun.security.krb5.internal.SeqNumber;

import java.io.Writer;
import java.nio.file.Path;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

import static java.lang.Math.exp;

public interface Embedding<T> extends Function<T, Vec>, Metric<T> {
  @Nullable
  Vec apply(T to);

  interface Builder<T> {
    Builder<T> file(Path path);
    Builder<T> minWordCount(int count);
    Builder<T> window(WindowType type, int left, int right);
    Builder<T> iterations(int count);
    Builder<T> step(double step);

    Embedding<T> build();
  }

  enum Type {
    GLOVE(GloVeBuilder.class),
    DECOMP(DecompBuilder.class),
    MULTI_DECOMP(MultiDecompBuilder.class),
    ;

    private final Class<? extends Builder> builderClass;

    Type(Class<? extends Builder> builderClass) {
      this.builderClass = builderClass;
    }
  }

  enum WindowType {
    LINEAR(d -> 1./Math.abs(d)),
    FIXED(d -> 1.),
    EXP(d -> exp(-1e-1 * Math.abs(d))),
    ;
    private final IntToDoubleFunction weight;

    WindowType(IntToDoubleFunction weight) {
      this.weight = weight;
    }

    public double weight(int dist) {
      return weight.applyAsDouble(dist);
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
