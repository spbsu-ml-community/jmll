package com.expleague.embedding.evaluation;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class QualityMetric {
  protected final EmbeddingImpl<CharSeq> embedding;
  protected List<List<CharSeq>> metrics = new ArrayList<>();
  protected List<String> files = new ArrayList<>();
  protected int metricsNumber = 0;

  public QualityMetric(EmbeddingImpl<CharSeq> embedding) {
    this.embedding = embedding;
  }

  protected abstract void check(List<String> wordsLine, int lineNumber) throws IOException;

  public abstract void measure(String input, String output) throws IOException;

  private CharSeq normalizeWord(String input) {
    return CharSeq.copy(input.toLowerCase());
  }

  protected void readMetricsNames(String fileName) throws IOException {
    File file = new File(fileName);
    BufferedReader fin;
    try {
      fin = new BufferedReader(new FileReader(file));
    } catch (FileNotFoundException e) {
      throw new IOException("Couldn't find the file with names of metrics files.");
    }
    try {
      int num = Integer.parseInt(fin.readLine());
      for (int i = 0; i < num; i++)
        files.add(fin.readLine());
      fin.close();
    } catch (IOException e) {
      throw new IOException("Error occurred during reading from the file.");
    }
  }

  protected String readMetricsFile(String input) throws IOException {
    metrics = new ArrayList<>();
    File file = new File(input);
    BufferedReader fin;
    try {
      fin = new BufferedReader(new FileReader(file));
    } catch (IOException e) {
      throw new IOException("Couldn't find the file to readMetricsFile metrics from: " + file);
    }
    try {
      metricsNumber = Integer.parseInt(fin.readLine());
      for (int i = 0; i < metricsNumber; i++) {
        List<String> line = Arrays.asList(fin.readLine().split(" "));
        check(line, i);
        List<CharSeq> leline = new ArrayList<>();
        for (String word: line) {
          leline.add(CharSeq.copy(normalizeWord(word)));
        }
        metrics.add(leline);
      }
      fin.close();
    } catch (IOException e) {
      throw new IOException("Error occurred during reading from the file.");
    }
    return file.getName();
  }

  protected boolean isWordsListInVocab(List<CharSeq> words) {
    for (CharSeq word: words) {
      if (!embedding.inVocab(word))
        return false;
    }
    return true;
  }

  public List<CharSeq> getClosestWordsExcept(Vec vector, int top, List<CharSeq> exceptWords) {
    int[] order = ArrayTools.sequence(0, embedding.vocabSize());
    TIntSet exceptIds = new TIntHashSet();
    exceptWords.forEach(word -> exceptIds.add(embedding.getIndex(word)));
    double[] weights = IntStream.of(order).parallel().mapToDouble(idx -> {
      if (exceptIds.contains(idx))
        return Double.MAX_VALUE;
      return -VecTools.cosine(embedding.apply(embedding.getObj(idx)), vector);
    }).toArray();
    ArrayTools.parallelSort(weights, order);
    return IntStream.range(0, top).mapToObj(idx ->
        embedding.getObj(order[idx])).collect(Collectors.toList());
  }

  public int getIndex(CharSeq word) {
    return embedding.getIndex(word);
  }

  public List<CharSeq> getClosestWords(CharSeq word, int top) {
    if (embedding.inVocab(word)) {
      final Vec vector = embedding.apply(word);
      return getClosestWordsExcept(vector, top, new ArrayList<>());
    } else {
      return null;
    }

  }
}
