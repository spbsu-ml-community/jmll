package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.optimization.impl.SAGADescent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class RunImdb {
  private static final int ALPHABET_SIZE = 7500;
  private static final int TRAIN_SIZE = 25000;
  private static final FastRandom random = new FastRandom(239);

  private static final int BOOST_ITERS = 1000;
  private static final double BOOST_STEP = 0.3;
  private static final int MAX_STATE_COUNT = 4;
  private static final int DESCENT_STEP_COUNT = TRAIN_SIZE * 160 + 100;
  private static final double GRAD_STEP = 0.5;
  private static final int THREAD_COUNT = 1;

  private List<Seq<Integer>> train;
  private Vec trainTarget;

  private List<Seq<Integer>> test;
  private Vec testTarget;
  private final List<Character> alphabet = new ArrayList<>();
  private int maxLen;

  private Dictionary<Character> dictionary;

  public void loadData() throws IOException {
    System.out.println("Number of cores: " + Runtime.getRuntime().availableProcessors());
    System.out.println("Alphabet size: " + ALPHABET_SIZE);
    System.out.println("States count: " + MAX_STATE_COUNT);
    System.out.println("GradBoost step: " + BOOST_STEP);
    System.out.println("GradBoost iters: " + BOOST_ITERS);
    System.out.println("GradDesc step: " + GRAD_STEP);
    System.out.println("Grad iters: " + DESCENT_STEP_COUNT);
    System.out.println("Train size: " + TRAIN_SIZE);


    List<CharSeq> positiveRaw = readData("src/aclImdb/train/pos");
    List<CharSeq> negativeRaw = readData("src/aclImdb/train/neg");

    List<CharSeq> all = new ArrayList<>(positiveRaw);
    all.addAll(negativeRaw);
    DictExpansion<Character> de = new DictExpansion<>(all.stream().flatMapToInt(CharSequence::chars)
        .sorted()
        .distinct()
        .mapToObj(i -> (char) i)
        .collect(Collectors.toList()), ALPHABET_SIZE);
    for (int i = 0; i < 10; i++) {
      positiveRaw.forEach(de::accept);
      negativeRaw.forEach(de::accept);
    }
    dictionary = de.result();
    //System.out.println("New dictionary: " + result.alphabet().toString());
    System.out.println("New dictionary size: " + dictionary.alphabet().size());

    int size = 0;
    for (CharSeq seq: positiveRaw) {
      size += dictionary.parse(seq).length();
    }
    System.out.println(size + " " + size / positiveRaw.size());
    System.out.println("Real alphabet size = " + dictionary.size());
    Collections.shuffle(positiveRaw, random);
    Collections.shuffle(negativeRaw, random);
    positiveRaw = positiveRaw.stream().limit(TRAIN_SIZE).collect(Collectors.toList());
    negativeRaw = negativeRaw.stream().limit(TRAIN_SIZE).collect(Collectors.toList());

    train = positiveRaw.stream().map(dictionary::parse).collect(Collectors.toList());
    train.addAll(negativeRaw.stream().map(dictionary::parse).collect(Collectors.toList()));
    maxLen = 0;
    for (int i = 0; i < train.size(); i++) {
      maxLen = Math.max(maxLen, train.get(i).length());
    }

    int[] targetArray = new int[train.size()];
    for (int i = 0; i < train.size() / 2; i++) {
      targetArray[i] = 1;
    }
    for (int i = train.size() / 2; i < train.size(); i++) {
      targetArray[i] = 0;
    }
    trainTarget = VecTools.fromIntSeq(new IntSeq(targetArray));

    test = readData("src/aclImdb/test/pos").stream().map(dictionary::parse).collect(Collectors.toList());
    test.addAll(readData("src/aclImdb/test/neg").stream().map(dictionary::parse).collect(Collectors.toList()));
    for (int i = 0; i < train.size(); i++) {
      maxLen = Math.max(maxLen, train.get(i).length());
    }
    targetArray = new int[test.size()];
    for (int i = 0; i < test.size() / 2; i++) {
      targetArray[i] = 1;
    }
    for (int i = test.size() / 2; i < test.size(); i++) {
      targetArray[i] = 0;
    }
    testTarget = VecTools.fromIntSeq(new IntSeq(targetArray));
    System.out.println("Data loaded");
  }

  public void test() {

    DataSet<Seq<Integer>> data = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return train.get(i);
      }

      @Override
      public int length() {
        return train.size();
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    };


    final GradientSeqBoosting<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
        new BootstrapSeqOptimization<>(
            new PNFA<>(MAX_STATE_COUNT, ALPHABET_SIZE, random, new SAGADescent(
                GRAD_STEP, DESCENT_STEP_COUNT, random, THREAD_COUNT
            )), random
        ), BOOST_ITERS, BOOST_STEP
    );


    Action<Computable<Seq<Integer>, Vec>> listener = classifier -> {
      System.out.println("Current time: " + new SimpleDateFormat("yyyy/MM/dd_HH:mm:ss").format(Calendar.getInstance().getTime()));
      System.out.println("Current accuracy:");
      System.out.println("Train accuracy: " + getAccuracy(train, trainTarget, classifier));
      System.out.println("Test accuracy: " + getAccuracy(test, testTarget, classifier));
      System.out.println("Train loss: " + getLoss(train, trainTarget, classifier));
      System.out.println("Test loss: " + getLoss(test, testTarget, classifier));
    };

    boosting.addListener(listener);
    final Computable<Seq<Integer>, Vec> classifier = boosting.fit(data, new LLLogit(trainTarget, null));

    System.out.println("Train accuracy: " + getAccuracy(train, trainTarget, classifier));
    System.out.println("Test accuracy: " + getAccuracy(test, testTarget, classifier));
  }

  private List<CharSeq> readData(final String filePath) throws IOException {
    long start = System.nanoTime();
    final List<CharSeq> data = Files.list(Paths.get(filePath)).map(path -> {
      try {
        return new CharSeqArray(Files.readAllLines(path)
            .stream()
            .map(String::toLowerCase)
            .map(str -> str.replaceAll("[^\\x00-\\x7F]", "").replaceAll("\\s{2,}", " ").trim())
            .collect(Collectors.joining("\n"))
            .toCharArray());
      } catch (IOException e) {
        e.printStackTrace();
        return null;
      }
    }).filter(x -> x != null).collect(Collectors.toList());
    System.out.printf("Data read in %.2f minutes\n", (System.nanoTime() - start) / 60e9);
    return data;
  }

  private double getAccuracy(List<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> classifier) {
    int passedCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      final double val = classifier.compute(data.get(i)).get(0);
      if ((target.get(i) > 0 && val > 0) || (target.get(i) <= 0 && val <= 0)) {
        passedCnt++;
      }
    }
    return 1.0 * passedCnt / data.size();
  }

  private double getLoss(List<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> classifier) {
    final LLLogit lllogit = new LLLogit(target, null);
    Vec values = new ArrayVec(target.dim());
    for (int i =0 ; i < target.dim(); i++) {
      values.set(i, classifier.compute(data.get(i)).get(0));
    }
    return lllogit.value(values);
  }

  public static void main(String[] args) throws IOException {
    RunImdb test = new RunImdb();
    test.loadData();
    test.test();
  }

}
