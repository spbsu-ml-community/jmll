package com.expleague.embedding.evaluation.metrics;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.LM.LWMatrixMultBuilder;
import gnu.trove.map.TObjectIntMap;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class SentenceGenerationMetric {
  static private LWMatrixMultBuilder embedding;
  final static private int dim = 5;
  static private Mx C0;

  public static void main(String[] args) throws IOException {
    String file = "/home/katyakos/diploma/proj6_spbau/data/tests/sentences/all_metrics_files.txt";

    embedding = (LWMatrixMultBuilder) Embedding.builder(Embedding.Type.LIGHT_WEIGHT_MATRIX_MULT);
    embedding
        .dim(dim)
        .minWordCount(1)
        .iterations(50)
        .step(0.5)
        .window(Embedding.WindowType.LINEAR, 10, 10);

    C0 = embedding.C0();

    List<String> files = readMetricsNames(file);
    for (String fileName : files) {
      final List<CharSeq> sentence = readMetricsFile(fileName);
      System.out.println("Started working with " + fileName);

      embedding.file(Paths.get(fileName)).build();
      measure(sentence);
    }

  }

  private static CharSeq normalizeWord(String input) {
    return CharSeq.copy(input.toLowerCase());
  }

  private static void measure(List<CharSeq> sentenceWords) {
    final List<CharSeq> wordsList = embedding.getVocab();
    final TObjectIntMap<CharSeq>  wordToIndex = embedding.getWords();
    final int vocabSize = wordsList.size();
    final int sentSize = sentenceWords.size();
    final int firstWord = wordToIndex.get(sentenceWords.get(0));

    Mx C = MxTools.multiply(C0, embedding.getContextMat(firstWord));
    StringBuilder result = (new StringBuilder()).append(sentenceWords.get(0)).append(" ");

    for (int t = 1; t < sentSize; t++) {
      int[] order = ArrayTools.sequence(0, vocabSize);
      final Mx Ctmp = VecTools.copy(C);
      double[] weights = IntStream.of(order).parallel().mapToDouble(idx -> -embedding.getProbability(Ctmp, idx)).toArray();
      ArrayTools.parallelSort(weights, order);
      final int newWord = order[0];

      C = MxTools.multiply(C, embedding.getContextMat(newWord));
      result.append(wordsList.get(newWord)).append(" ");
    }

    System.out.println("Original sentence is:\t" + String.valueOf(sentenceWords));
    System.out.println("Generated sentence is:\t" + result.toString());
  }

  private static List<String> readMetricsNames(String fileName) throws IOException {
    File file = new File(fileName);
    BufferedReader fin;
    try {
      fin = new BufferedReader(new FileReader(file));
    } catch (FileNotFoundException e) {
      throw new IOException("Couldn't find the file with names of metrics files.");
    }
    try {
      List<String> files = new ArrayList<>();
      int num = Integer.parseInt(fin.readLine());
      for (int i = 0; i < num; i++)
        files.add(fin.readLine());
      fin.close();
      return files;
    } catch (IOException e) {
      throw new IOException("Error occurred during reading from the file.");
    }
  }

  private static List<CharSeq> readMetricsFile(String input) throws IOException {
    File file = new File(input);
    BufferedReader fin;
    try {
      fin = new BufferedReader(new FileReader(file));
    } catch (IOException e) {
      throw new IOException("Couldn't find the file to readMetricsFile metrics from: " + file);
    }
    try {
      String[] line = fin.readLine().split(" ");
      List<CharSeq> leline = new ArrayList<>();
      for (String word: line) {
        leline.add(CharSeq.copy(normalizeWord(word)));
      }
      fin.close();
      return leline;
    } catch (IOException e) {
      throw new IOException("Error occurred during reading from the file.");
    }
  }
}
