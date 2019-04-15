package com.expleague.embedding;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.embedding.evaluation.QualityMetric;
import com.expleague.embedding.evaluation.metrics.CloserFurtherMetric;
import com.expleague.embedding.evaluation.metrics.WordAnalogiesMetric;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.glove.GloVeBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.kmeans.ClusterBasedSymmetricBuilder;
import javafx.util.Pair;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {
    /*Embedding.Builder builder = Embedding.builder(Embedding.Type.HIERARCH_CLUSTER);
    String file = args[0];
    final Embedding result = builder
//        .dim(50)
        .iterations(5)
        .step(5e-2)
//        .minWordCount(1)
        .window(Embedding.WindowType.LINEAR, 7, 7)
        .file(Paths.get(file))
        .build();*/
    GloVeBuilder builder = (GloVeBuilder) Embedding.builder(Embedding.Type.GLOVE);
    String file = args[0];
    final Embedding result = builder
        .dim(50)
        .minWordCount(5)
        .iterations(25)
        .step(0.05)
        .window(Embedding.WindowType.LINEAR, 15, 15)
        .file(Paths.get(file))
        .build();
    try (Writer to = Files.newBufferedWriter(Paths.get(StreamTools.stripExtension(file) + ".ss_decomp"))) {
      Embedding.write(result, to);
    }

    /*String input = "/home/katyakos/diploma/proj6_spbau/data/corpuses/text8";
    String res = "/home/katyakos/diploma/proj6_spbau/data/models/text8/shrinking/";
    String metricNames = "/home/katyakos/diploma/proj6_spbau/data/tests/text8/all_metrics_files.txt";
    String resultDec = "/home/katyakos/diploma/proj6_spbau/data/tests/text8/results_decomp/shrinking/";
    String resultGl = "/home/katyakos/diploma/proj6_spbau/data/tests/text8/results_glove/shrinking/";
    int[] glove = {25, 50, 80, 120};
    int[] glove_iters = {25};
    int[][] decomp = {{40, 5},{40, 10}, {50, 5}, {50, 10}, {50, 20}, {80, 50}, {80, 10},  {80, 20}, {120, 10}, {120, 20}};
    int[] decomp_iters = {};

    for (int gl : glove) {
      for (int it : glove_iters) {
        GloVeBuilder builder = (GloVeBuilder) Embedding.builder(Embedding.Type.GLOVE);
        Interval.start();
        final Embedding result = builder
            .dim(gl)
            .minWordCount(5)
            .iterations(it)
            .step(0.05)
            .window(Embedding.WindowType.LINEAR, 15, 15)
            .file(Paths.get(input))
            .build();
        Interval.stopAndPrint("Glove Dim " + String.valueOf(gl) + " It " + String.valueOf(it));
        String fileResult = res + "/glove/glove-" + String.valueOf(gl) + "-" + String.valueOf(it);
        if (!Files.exists(Paths.get(fileResult)))
          Files.createFile(Paths.get(fileResult));
        try (Writer to = Files.newBufferedWriter(Paths.get(fileResult))) {
          Embedding.write(result, to);
        }

        try (Reader from = Files.newBufferedReader(Paths.get(fileResult))) {
          final EmbeddingImpl embedding = EmbeddingImpl.read(from, CharSeq.class);
          WordAnalogiesMetric metric = new WordAnalogiesMetric(embedding);
          String metricResult = resultGl + "glove-" + String.valueOf(gl) + "-" + String.valueOf(it);
          if (!Files.exists(Paths.get(metricResult)))
            Files.createDirectory(Paths.get(metricResult));
          metric.measure(metricNames, metricResult);
        }
      }
    }

    for (int[] dec : decomp) {
      int sym = dec[0], skew = dec[1];
      for (int it : decomp_iters) {
        DecompBuilder builder = (DecompBuilder) Embedding.builder(Embedding.Type.DECOMP);
        Interval.start();
        final Embedding result = builder
            .dimSym(sym).dimSkew(skew)
            .minWordCount(5)
            .iterations(it)
            .step(0.05)
            .window(Embedding.WindowType.LINEAR, 15, 15)
            .file(Paths.get(input))
            .build();
        Interval.stopAndPrint("Decomp Sym " + String.valueOf(sym) + " Skew" + String.valueOf(skew) + " It " + String.valueOf(it));
        String fileResult = res + "/decomp/decomp-" + String.valueOf(sym) + "-" + String.valueOf(skew) + "-" + String.valueOf(it);
        if (!Files.exists(Paths.get(fileResult)))
          Files.createFile(Paths.get(fileResult));
        try (Writer to = Files.newBufferedWriter(Paths.get(fileResult))) {
          Embedding.write(result, to);
        }

        try (Reader from = Files.newBufferedReader(Paths.get(fileResult))) {
          final EmbeddingImpl embedding = EmbeddingImpl.read(from, CharSeq.class);
          WordAnalogiesMetric metric = new WordAnalogiesMetric(embedding);
          String metricResult = resultDec + "decomp-" + String.valueOf(sym) + "-" + String.valueOf(skew) + "-" + String.valueOf(it);
          if (!Files.exists(Paths.get(metricResult)))
            Files.createDirectory(Paths.get(metricResult));
          metric.measure(metricNames, metricResult);
        }
      }
    }*/

  }
}
