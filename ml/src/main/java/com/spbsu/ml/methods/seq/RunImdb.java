package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.seq.nn.*;
import com.spbsu.ml.optimization.impl.SAGADescent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
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
  private static final boolean PRINT_DEBUG = true;
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
    /*
    Writer writer = new FileWriter("data.txt");
    writer.write(train.size() + "\n");
    for (int i = 0; i < train.size(); i++) {
      writer.write(train.get(i) + "\n");
    }
    writer.write(trainTarget.toString());

    writer.write(test.size() + "\n");
    for (int i = 0; i < test.size(); i++) {
      writer.write(test.get(i) + "\n");
    }
    writer.write(test.toString());
    */
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


    long start = System.nanoTime();
    final GradientSeqBoosting<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
        new BootstrapSeqOptimization<>(
            new PNFA<>(MAX_STATE_COUNT, ALPHABET_SIZE, random, new SAGADescent(
                GRAD_STEP, DESCENT_STEP_COUNT, random, THREAD_COUNT
            )), random
        ), BOOST_ITERS, BOOST_STEP
    );

//    GradientSeqBoosting<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
//        new BootstrapSeqOptimization<>(
//            new PNFANetworkSAGA<>(
//                new IntAlphabet(ALPHABET_SIZE),
//                MAX_STATE_COUNT,
//                random,
//                GRAD_STEP,
//                DESCENT_STEP_COUNT,
//                dictionary.alphabet(),
//                PRINT_DEBUG
//                //4
//            ),
//            random),
//        //new IncrementalAutomatonBuilder<>(new IntAlphabet(ALPHABET_SIZE), new OptimizedStateEvaluation<>(), MAX_STATE_COUNT, 1000000),
//        BOOST_ITERS,
//        BOOST_STEP
//    );
    Action<Computable<Seq<Integer>, Vec>> listener = classifier -> {
      System.out.println("Current time: " + new SimpleDateFormat("yyyy/MM/dd_HH:mm:ss").format(Calendar.getInstance().getTime()));
      System.out.println("Current accuracy:");
      System.out.println("Train accuracy: " + getAccuracy(train, trainTarget, classifier));
      System.out.println("Test accuracy: " + getAccuracy(test, testTarget, classifier));
      System.out.println("Train evaluation: " + getLoss(train, trainTarget, classifier));
      System.out.println("Test evaluation: " + getLoss(test, testTarget, classifier));
    };

    boosting.addListener(listener);
    final Computable<Seq<Integer>, Vec> classifier = boosting.fit(data, new LLLogit(trainTarget, null));

/*
    final int signalDim = ALPHABET_SIZE;
    final int lstmNodeCount = 60;
    final int logisticNodeCount = 1;

    final NeuralNetwork network = new NeuralNetwork(
            new LSTMLayer(lstmNodeCount, signalDim, random),
            new MeanPoolLayer(),
            new LogisticLayer(logisticNodeCount, lstmNodeCount, random)
    );

    final FuncC1[] targetFuncs = new FuncC1[trainTarget.dim()];
    for (int i = 0;i  < trainTarget.dim(); i++) {
      final int i1 = i;
      final Vec v = new ArrayVec(1);
      final L2 llLogit = new L2(v, data);
      v.set(0, trainTarget.get(i));
      targetFuncs[i] = new FuncC1.Stub() {
        @Override
        public Vec gradient(Vec x) {
          network.setParams(x);
          final Mx mx = seqToMx(train.get(i1));
          final Vec outputGrad = llLogit.gradient(network.value(mx));
          final Mx outputGradMx = new VecBasedMx(1, outputGrad.dim());
          for (int i = 0; i < outputGrad.dim(); i++) {
            outputGradMx.set(i, outputGrad.get(i));
          }
          return network.gradByParams(mx, outputGradMx, true);
        }

        @Override
        public double value(Vec x) {
          network.setParams(x);
          final Mx mx = seqToMx(train.get(i1));
          return llLogit.value(network.value(mx));
        }

        @Override
        public int dim() {
          return network.paramCount();
        }
      };
    }

    final FuncEnsemble networkTarget = new FuncEnsemble<>(Arrays.asList(targetFuncs), GRAD_STEP);
    final Vec optW = new SAGA(10, GRAD_STEP, random).optimize(networkTarget);
    network.setParams(optW);

    //System.out.printf("grad step=%.4f, boost step=%.4f, boost iters=%d, elapsed %.2f minutes\n", GRAD_STEP, BOOST_STEP, BOOST_ITERS, (System.nanoTime() - start) / 60e9);
    final Computable<Mx, Vec> classifier = network::value;
*/
    System.out.println("Train accuracy: " + getAccuracy(train, trainTarget, classifier));
    System.out.println("Test accuracy: " + getAccuracy(test, testTarget, classifier));

//    System.out.println("Train accuracy: " + getAccuracyMx(train, trainTarget, classifier));
//    System.out.println("Test accuracy: " + getAccuracyMx(test, testTarget, classifier));
  }

  private List<CharSeq> readData(final String filePath) throws IOException {
    //final List<CharSeq> data = new ArrayList<>();

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

  private double getAccuracyMx(List<Seq<Integer>> data, Vec labels, Computable<Mx, Vec> classifier) {
    int passedCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      if (Math.round(classifier.compute(seqToMx(data.get(i))).get(0)) == Math.round(labels.get(i))) {
        passedCnt++;
      }
    }
    return 1.0 * passedCnt / data.size();
  }

  private double getLoss(List<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> classifier) {
    final LLLogit lllogit = new LLLogit(target, null);
    double result = 0;
    Vec values = new ArrayVec(target.dim());
    for (int i =0 ; i < target.dim(); i++) {
      values.set(i, classifier.compute(data.get(i)).get(0));
    }
    return lllogit.value(values);
  }

  private Mx seqToMx(Seq<Integer> seq) {
    final Mx mx = new VecBasedMx(maxLen, ALPHABET_SIZE);
    for (int j = 0; j < Math.min(maxLen, seq.length()); j++) {
      mx.set(j, seq.at(j), 1);
    }
    for (int j = Math.min(maxLen, seq.length()); j < maxLen; j++) {
      mx.set(j, 0, 1);
    }
    return mx;
  }

  public static void main(String[] args) throws IOException {
    RunImdb test = new RunImdb();
    test.loadData();
    test.test();
  }

}
