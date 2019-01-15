package com.expleague.ml.embedding.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.embedding.Embedding;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;

public class EmbeddingImpl<T> implements Embedding<T> {
  private final Map<T, Vec> mapping;

  public EmbeddingImpl(Map<T, Vec> mapping) {
    this.mapping = mapping;
  }

  @Override
  public double distance(T a, T b) {
    Vec vA = mapping.get(a);
    Vec vB = mapping.get(b);
    return vA == null || vB == null ? Double.POSITIVE_INFINITY : VecTools.cosine(vA, vB);
  }

  @Override
  public Vec apply(T t) {
    return mapping.get(t);
  }

  public void write(Writer to) {
    mapping.forEach((word, vec) -> {
      try {
        to.append(DataTools.SERIALIZATION.write(word))
            .append('\t')
            .append(DataTools.SERIALIZATION.write(vec))
            .append('\n');
      }
      catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  public static <T> EmbeddingImpl<T> read(Reader from, Class<? extends T> clazz) {
    BufferedReader bufferedReader = new BufferedReader(from);

    Map<T, Vec> mapping = new HashMap<>();
    bufferedReader.lines().forEach(line -> {
      int partitionIdx = line.lastIndexOf('\t');
      T word = DataTools.SERIALIZATION.read(line.substring(0, partitionIdx), clazz);
      Vec vec = DataTools.SERIALIZATION.read(line.substring(partitionIdx + 1), ArrayVec.class);
      mapping.put(word, vec);
    });

    return new EmbeddingImpl<>(mapping);
  }
}
