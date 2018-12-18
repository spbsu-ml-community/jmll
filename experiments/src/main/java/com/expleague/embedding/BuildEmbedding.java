package com.expleague.embedding;

import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {
//    final Embedding result = Embedding.builder(Embedding.Type.DECOMP).file(Paths.get("/Users/solar/tree/proj6_spbau/data/corpuses/text8")).build();
    DecompBuilder builder = (DecompBuilder)Embedding.builder(Embedding.Type.DECOMP);
    final Embedding result = builder
        .dimSym(130)
        .dimSkew(20)
        .file(Paths.get("/Users/solar/tree/proj6_spbau/data/corpuses/lenta.txt"))
//        .file(Paths.get("/Users/solar/tree/proj6_spbau/data/corpuses/text8"))
        .build();
    Embedding.write(result, Files.newBufferedWriter(Paths.get("/Users/solar/tree/proj6_spbau/data/corpuses/lenta.decomp")));
  }
}
