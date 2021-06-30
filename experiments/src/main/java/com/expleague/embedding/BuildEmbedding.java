package com.expleague.embedding;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.seq.CharSeq;
import com.expleague.embedding.evaluation.metrics.CloserFurtherMetric;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.lm.LWMatrixMultBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BuildEmbedding {
  public static void main(String[] args) throws IOException {

    String file = args[0];
    String embeddingFile = StreamTools.stripExtension(file) + ".lwmm";
    String resultFile = "/Users/solar/data/text/sentences/shot.txt";
    String metricFile = "/home/katyakos/diploma/proj6_spbau/data/tests/hobbit/all_metrics_files.txt";

    LWMatrixMultBuilder builder = (LWMatrixMultBuilder) Embedding.builder(Embedding.Type.LIGHT_WEIGHT_MATRIX_MULT);
    final Embedding result = builder
        .dim(10)
        .minWordCount(1)
        .iterations(15)
        .step(0.05)
        .window(Embedding.WindowType.LINEAR, 5, 5)
        .file(Paths.get(file))
        .build();
    try (Writer to = Files.newBufferedWriter(Paths.get(embeddingFile))) {
      Embedding.write(result, to);
    }

    try (Reader from = Files.newBufferedReader(Paths.get(embeddingFile))) {
      final EmbeddingImpl embedding = EmbeddingImpl.read(from, CharSeq.class);
      CloserFurtherMetric metric = new CloserFurtherMetric(embedding);
      if (!Files.exists(Paths.get(resultFile)))
        Files.createDirectory(Paths.get(resultFile));
      metric.measure(metricFile, resultFile);
    }


    /*NgramGloveBuilder glove_builder = (NgramGloveBuilder) Embedding.builder(Embedding.Type.NGRAM_GLOVE);
    String file = args[0];
    final Embedding result = glove_builder
        .dim(150)
        .minWordCount(5)
        .iterations(25)
        .step(0.05)
        .window(Embedding.WindowType.LINEAR, 15, 15)
        .file(Paths.get(file))
        .build();
    try (Writer to = Files.newBufferedWriter(Paths.get(StreamTools.stripExtension(file) + ".glove"))) {
      Embedding.write(result, to);
    }*/

    /*String input = "/home/katyakos/diploma/proj6_spbau/data/corpuses/corpus2Gb_parts/";
    String res = "/home/katyakos/diploma/proj6_spbau/data/models/corpus2Gb/";
    String metricNames = "/home/katyakos/diploma/proj6_spbau/data/tests/text8/all_metrics_files.txt";
    String resultDec = "/home/katyakos/diploma/proj6_spbau/data/tests/corpus2Gb/results_decomp/";
    String resultGl = "/home/katyakos/diploma/proj6_spbau/data/tests/corpus2Gb/results_glove/";
    int[] glove = {25};
    int[] glove_iters = {25};
    int[][] decomp = {{25, 5}, {25, 15}, {25, 20}, {50, 10}, {50, 20}, {50, 40}, {80, 10}, {80, 30}, {80, 50}, {80, 70}};
    int[] decomp_iters = {};

    for (int corp_num = 1; corp_num < 2; corp_num++) {
      String inputFile = input + "corpus2Gb_" + String.valueOf(corp_num);
      for (int gl : glove) {
        for (int it : glove_iters) {
          GloVeBuilder builder = (GloVeBuilder) Embedding.builder(Embedding.Type.GLOVE);
          Interval.start();
          final Embedding result =
              builder.dim(gl).minWordCount(5).iterations(it)
                  .step(0.05).window(Embedding.WindowType.FIXED, 15, 15)
                  .file(Paths.get(inputFile)).build();
          Interval.stopAndPrint("Corpus number " + String.valueOf(corp_num) + " Glove Dim " + String.valueOf(gl) + " It " + String.valueOf(it));
          String fileResult = res + String.valueOf(corp_num) + "/glove/glove-" + String.valueOf(gl) + "-" + String.valueOf(it);
          if (!Files.exists(Paths.get(fileResult)))
            Files.createFile(Paths.get(fileResult));
          try (Writer to = Files.newBufferedWriter(Paths.get(fileResult))) {
            Embedding.write(result, to);
          }

          try (Reader from = Files.newBufferedReader(Paths.get(fileResult))) {
            final EmbeddingImpl embedding = EmbeddingImpl.read(from, CharSeq.class);
            WordAnalogiesMetric metric = new WordAnalogiesMetric(embedding);
            String metricResult = resultGl + String.valueOf(corp_num) + "/glove-" + String.valueOf(gl) + "-" + String.valueOf(it);
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
          final Embedding result =
              builder
                  .regularization(false).dimSym(sym).dimSkew(skew)
                  .minWordCount(5).iterations(it).step(0.1)
                  .window(Embedding.WindowType.LINEAR, 15, 15)
                  .file(Paths.get(inputFile)).build();
          Interval.stopAndPrint("Corpus number " + String.valueOf(corp_num) + " Decomp Sym " + String.valueOf(sym) + " Skew" + String.valueOf(skew) + " It " + String.valueOf(it));
          String fileResult = res + String.valueOf(corp_num) + "/decomp/decomp-" + String.valueOf(sym) + "-" + String.valueOf(skew) + "-" + String.valueOf(it);
          if (!Files.exists(Paths.get(fileResult)))
            Files.createFile(Paths.get(fileResult));
          try (Writer to = Files.newBufferedWriter(Paths.get(fileResult))) {
            Embedding.write(result, to);
          }

          try (Reader from = Files.newBufferedReader(Paths.get(fileResult))) {
            final EmbeddingImpl embedding = EmbeddingImpl.read(from, CharSeq.class);
            WordAnalogiesMetric metric = new WordAnalogiesMetric(embedding);
            String metricResult = resultDec + String.valueOf(corp_num) + "/decomp-" + String.valueOf(sym) + "-" + String.valueOf(skew) + "-" + String.valueOf(it);
            if (!Files.exists(Paths.get(metricResult)))
              Files.createDirectory(Paths.get(metricResult));
            metric.measure(metricNames, metricResult);
          }
        }
      }
    }*/

  }
}
