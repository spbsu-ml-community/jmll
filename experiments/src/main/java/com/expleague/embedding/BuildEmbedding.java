package com.expleague.embedding;

import com.expleague.commons.io.StreamTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {
    DecompBuilder builder = (DecompBuilder)Embedding.builder(Embedding.Type.DECOMP);
    String file = args[0];
    final Embedding result = builder
        .dimSym(100)
        .dimSkew(10)
        .window(Embedding.WindowType.FIXED, 3, 5)
        .file(Paths.get(file))
        .build();
    try (Writer to = Files.newBufferedWriter(Paths.get(StreamTools.stripExtension(file) + ".decomp"))) {
      Embedding.write(result, to);
    }
  }
}
