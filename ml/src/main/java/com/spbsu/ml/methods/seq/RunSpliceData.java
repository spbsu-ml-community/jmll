package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.SeqOptimization;
import com.spbsu.ml.methods.seq.nn.PNFANetworkGD;
import com.spbsu.ml.methods.seq.nn.PNFANetworkSAGA;
import com.spbsu.ml.methods.seq.nn.PNFANetworkSGD;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class RunSpliceData {
  private static final List<String> CLASSES = Arrays.asList("EI", "IE", "N");
  private static final int ALPHABET_SIZE = 10;
  private static final int BOOST_ITERS = 30000;
  private static final double BOOST_STEP = 0.1;
  private static final int MAX_STATE_COUNT = 8;
  private static final int DESCENT_STEP_COUNT = 100000;
  private static final double GRAD_STEP = 0.75;
  private static final FastRandom random = new FastRandom(239);

  final List<Seq<Integer>> trainData = new ArrayList<>();
  Vec trainTarget;

  final List<Seq<Integer>> testData = new ArrayList<>();
  Vec testTarget;

  final Alphabet<Integer> alphabet = new IntAlphabet(ALPHABET_SIZE);

  public void loadData() throws IOException {
    System.out.println("Number of cores: " + Runtime.getRuntime().availableProcessors());
    System.out.println("Alphabet size: " + ALPHABET_SIZE);
    System.out.println("States count: " + MAX_STATE_COUNT);
    System.out.println("GradBoost step: " + BOOST_STEP);
    System.out.println("GradBoost iters: " + BOOST_ITERS);
    System.out.println("GradDesc step: " + GRAD_STEP);
    System.out.println("Grad iters: " + DESCENT_STEP_COUNT);

    final List<CharSeq> data = new ArrayList<>();
    final TIntList classes = new TIntArrayList();
    final int[] classCount = new int[CLASSES.size()];

    new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(
            Paths.get("ml/src/splice.data.txt.gz").toFile())), StandardCharsets.UTF_8)
    ).lines().forEach(line -> {
      final String[] tokens = line.split(",");
      final int clazz = CLASSES.indexOf(tokens[0]);
      if (clazz == -1) {
        throw new IllegalStateException("Unknown class " + tokens[0]);
      }
      classes.add(clazz);
      classCount[clazz]++;
      data.add(new CharSeqArray(tokens[2].trim().toCharArray()));
    });

    List<Character> alpha = Arrays.asList('A', 'C', 'G', 'T');
    DictExpansion<Character> de = new DictExpansion<>(alpha, ALPHABET_SIZE);
    for (int i = 0; i < 500; i++) {
      data.forEach(de::accept);
    }

    Dictionary<Character> result = de.result();

    System.out.println("New dictionary: " + result.alphabet().toString());

    int[] freqs = new int[ALPHABET_SIZE];
    int[] sum = new int[1];
    data.forEach(word -> {
      Seq<Integer> seq = result.parse(word);
      for (int i =0; i < seq.length(); i++) {
        freqs[seq.at(i)]++;
        sum[0]++;
      }
    });
    Map<String, Double> map = new TreeMap<>();
    for (int i = 0; i < ALPHABET_SIZE; i++) {
      map.put(result.get(i).toString(), 1.0 * freqs[i] / sum[0]);
    }

    map.entrySet().stream().sorted(Map.Entry.comparingByValue(Collections.reverseOrder())).forEach(it -> System.out.printf("%s: %.5f, ", it.getKey(), it.getValue()));
    System.out.println();
    final int sampleCount = Arrays.stream(classCount).min().orElse(0);

    final int trainCount = sampleCount * 8 / 10;
    int[] trainClasses = new int[trainCount * CLASSES.size()];
    int[] testClasses = new int[(sampleCount - trainCount) * CLASSES.size()];
    for (int clazz = 0; clazz < CLASSES.size(); clazz++) {
      int cnt = 0;
      for (int i = 0; i < data.size(); i++) {
        if (classes.get(i) != clazz) {
          continue;
        }
        if (cnt < trainCount) {
          trainClasses[trainData.size()] = classes.get(i);
          trainData.add(result.parse(data.get(i)));
        } else if (cnt < sampleCount){
          testClasses[testData.size()] = classes.get(i);
          testData.add(result.parse(data.get(i)));
        }
        cnt++;
      }
    }

    trainTarget = VecTools.fromIntSeq(new IntSeq(trainClasses));
    testTarget = VecTools.fromIntSeq(new IntSeq(testClasses));
  }

  public void test() {

    DataSet<Seq<Integer>> data = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return trainData.get(i);
      }

      @Override
      public int length() {
        return trainData.size();
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    };

    List<Integer> labels = new ArrayList<>();
    for (int i = 0; i < trainTarget.length(); i++) {
      labels.add((int) Math.round(trainTarget.get(i)));
    }
    long start = System.nanoTime();
    final GradientSeqBoosting<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
            //new IncrementalAutomatonBuilder<>(alphabet, new OptimizedStateEvaluation<>(), i),
            new PNFANetworkGD<>(alphabet, MAX_STATE_COUNT, random, GRAD_STEP, DESCENT_STEP_COUNT, 4),
            BOOST_ITERS, BOOST_STEP);

    Action<Computable<Seq<Integer>, Vec>> listener = classifier -> {
      System.out.println("Current time: " + new SimpleDateFormat("yyyyMMdd_HH:mm:ss").format(Calendar.getInstance().getTime()));
      // System.out.println("Current accuracy:");
      //     System.out.println("Train accuracy: " + getAccuracy(trainData, trainTarget, classifier));
      //   System.out.println("Test accuracy: " + getAccuracy(testData, testTarget, classifier));
//      System.out.println("Train evaluation: " + getLoss(trainData, trainTarget, classifier));
  //    System.out.println("Test evaluation: " + getLoss(testData, testTarget, classifier));
    };
    boosting.addListener(listener);

    final Computable<Seq<Integer>, Vec> classifier =
            new OneVsRest(boosting, CLASSES.size(), labels).fit(data, new LLLogit(trainTarget, null));
    long end = System.nanoTime();
    //.fit(data, new L2(trainTarget, null));
    //System.out.println(automaton.toString());
    System.out.println(String.format("Elapsed %.2f minutes", (end - start) / 60e9));
    System.out.println("Train accuracy of " + getAccuracy(trainData, trainTarget, classifier));
    System.out.println("Test accuracy of  " + getAccuracy(testData, testTarget, classifier));
  }

  private class OneVsRest implements SeqOptimization<Integer, LLLogit> {
    private final int classCount;
    private final SeqOptimization<Integer, LLLogit> optimization;
    private final List<Integer> labels;

    OneVsRest(SeqOptimization<Integer, LLLogit> optimization, int classCount, List<Integer> labels) {
      this.classCount = classCount;
      this.optimization = optimization;
      this.labels = labels;
    }

    @Override
    public Computable<Seq<Integer>, Vec> fit(DataSet<Seq<Integer>> learn, LLLogit llLogit) {
      List<Computable<Seq<Integer>, Vec>> classifiers = new ArrayList<>();
      for (int i = 0; i < classCount; i++) {
        Vec newTarget = VecTools.copy(llLogit.labels());
        for (int j = 0; j < llLogit.labels().length(); j++) {
          if (labels.get(j) != i) {
            newTarget.set(j, 0);
          } else {
            newTarget.set(j, 1);
          }
        }
        classifiers.add(optimization.fit(learn, new LLLogit(newTarget, null)));
      }

      return argument -> {
        double max = -Double.MAX_VALUE;
        int clazz = -1;
        for (int i = 0; i < classCount; i++) {
          double prob = classifiers.get(i).compute(argument).get(0);
          if (prob > max) {
            max = prob;
            clazz = i;
          }
        }
        return new SingleValueVec(clazz);
      };
    }
  }

  private double getLoss(List<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> classifier) {
    final LLLogit lllogit = new LLLogit(target, null);
    double result = 0;
    for (int i = 0; i < data.size(); i++) {
      result += lllogit.value(classifier.compute(data.get(i)));
    }
    return result / data.size();
  }

  private double getAccuracy(List<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> classifier) {
    int passedCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      if (Math.round(classifier.compute(data.get(i)).get(0)) == target.get(i)) {
        passedCnt++;
      }
    }
    return 1.0 * passedCnt / data.size();
  }

  public static void main(String[] args) throws IOException {
    RunSpliceData test = new RunSpliceData();
    test.loadData();
    test.test();
  }
}