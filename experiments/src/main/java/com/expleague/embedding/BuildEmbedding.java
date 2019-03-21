package com.expleague.embedding;

import com.expleague.commons.io.StreamTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.decomp.MultiDecompBuilder;
import com.expleague.ml.embedding.glove.GloVeBuilder;
import com.expleague.ml.embedding.kmeans.KmeansSkipBuilder;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {
    KmeansSkipBuilder builder = (KmeansSkipBuilder) Embedding.builder(Embedding.Type.KMEANS_SKIP);
    String file = args[0];
    final Embedding result = builder
        .dim(50)
        .iterations(25)
        .step(0.1)
//        .minWordCount(1)
        .window(Embedding.WindowType.FIXED, 15, 15)
        .file(Paths.get(file))
        .build();
    /*GloVeBuilder builder = (GloVeBuilder) Embedding.builder(Embedding.Type.GLOVE);
    String file = args[0];
    final Embedding result = builder
        .dim(50)
        .minWordCount(5)
        .iterations(25)
        .step(0.1)
        .window(Embedding.WindowType.LINEAR, 15, 15)
        .file(Paths.get(file))
        .build();*/
    try (Writer to = Files.newBufferedWriter(Paths.get(StreamTools.stripExtension(file) + ".ss_decomp"))) {
      Embedding.write(result, to);
    }
  }
}
