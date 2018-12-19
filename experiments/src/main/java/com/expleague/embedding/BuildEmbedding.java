package com.expleague.embedding;

import com.expleague.commons.io.StreamTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {
    DecompBuilder builder = (DecompBuilder)Embedding.builder(Embedding.Type.DECOMP);
    String file = args[0];
    final Embedding result = builder
        .dimSym(150)
        .dimSkew(20)
        .file(Paths.get(file))
        .build();
    Embedding.write(result, Files.newBufferedWriter(Paths.get(StreamTools.stripExtension(file) + ".decomp")));
  }
}
