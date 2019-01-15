package com.expleague.embedding;

import com.expleague.commons.io.StreamTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.MultiDecompBuilder;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {
    MultiDecompBuilder builder = (MultiDecompBuilder) Embedding.builder(Embedding.Type.MULTI_DECOMP);
    String file = args[0];
    final Embedding result = builder
        .dimSym(100)
        .dimSkew(10)
        .iterations(20)
        .step(0.05)
//        .minWordCount(1)
        .window(Embedding.WindowType.LINEAR, 15, 15)
        .file(Paths.get(file))
        .build();
    try (Writer to = Files.newBufferedWriter(Paths.get(StreamTools.stripExtension(file) + ".decomp"))) {
      Embedding.write(result, to);
    }
  }
}
