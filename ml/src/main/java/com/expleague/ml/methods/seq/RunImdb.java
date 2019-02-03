package com.expleague.ml.methods.seq;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.Dictionary;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqArray;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.methods.seq.param.*;
import com.expleague.ml.optimization.impl.AdamDescent;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RunImdb {
  private static final int ALPHABET_SIZE = 100;
  private static final int TRAIN_SIZE = 250;
  private static final FastRandom random = new FastRandom(239);

  private static final int BOOST_ITERS = 4;
  private static final int VALUES_EPOCH_COUNT = 5;
  private static final double VALUE_GRAD_STEP = 0.3;
  private final double alpha;
  private final double addToDiag;
  private final int stateCount;
  private final double boostStep;
  private final int epochCount;
  private final double gradientStep;
  private final String wParam;
  private final String bParam;
  private final boolean validateEachEpoch;

  private List<Seq<Integer>> train;
  private Vec trainTarget;

  private List<Seq<Integer>> test;
  private Vec testTarget;
  private final List<Character> alphabet = new ArrayList<>();
  private int maxLen;

  private Dictionary<Character> dictionary;

  private final BettaParametrization bettaParametrization;
  private final WeightParametrization weightParametrization;

  public RunImdb(final int stateCount, final double alpha, final double addToDiag, final double boostStep, int epochCount, double gradientStep, String wParam, String bParam, boolean validateEachEpoch) {
    this.stateCount = stateCount;
    this.alpha = alpha;
    this.addToDiag = addToDiag;
    this.boostStep = boostStep;
    this.epochCount = epochCount;
    this.gradientStep = gradientStep;
    this.wParam = wParam;
    this.bParam = bParam;
    this.validateEachEpoch = validateEachEpoch;
    if (bParam.equals("mx")) {
      bettaParametrization = new BettaMxParametrization(addToDiag);
    } else if (bParam.equals("vec")) {
      bettaParametrization = new BettaTwoVecParametrization(addToDiag);
    } else {
      throw new IllegalArgumentException(bParam);
    }

    if (wParam.equals("sqr")) {
      weightParametrization = new WeightSquareParametrization(bettaParametrization);
    } else if (wParam.equals("exp")) {
      weightParametrization = new WeightExpParametrization(bettaParametrization);
    } else {
      throw new IllegalArgumentException(wParam);
    }
  }

  public void loadWordData() throws IOException {
    train = new ArrayList<>(TRAIN_SIZE);
    test = new ArrayList<>(TRAIN_SIZE);
    trainTarget = new ArrayVec(TRAIN_SIZE);
    testTarget = new ArrayVec(TRAIN_SIZE);

    readWordData("src/train.txt", train, trainTarget);
    readWordData("src/test.txt", test, testTarget);
    //    loadData();
  }

  public void loadData() throws IOException {
    System.out.println("Number of cores: " + Runtime.getRuntime().availableProcessors());
    System.out.println("Alphabet size: " + ALPHABET_SIZE);
    System.out.println("States count: " + stateCount);
    System.out.println("GradBoost step: " + boostStep);
    System.out.println("GradBoost iters: " + BOOST_ITERS);
    System.out.println("GradDesc step: " + gradientStep);
    System.out.println("Grad iters: " + epochCount);
    System.out.println("Train size: " + TRAIN_SIZE);


    List<CharSeq> positiveRaw = readData("ml/src/aclImdb/train/pos");
    List<CharSeq> negativeRaw = readData("ml/src/aclImdb/train/neg");

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

    test = readData("ml/src/aclImdb/test/pos").stream().map(dictionary::parse).collect(Collectors
        .toList());
    test.addAll(readData("ml/src/aclImdb/test/neg").stream().map(dictionary::parse).collect
        (Collectors.toList()));
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


    IntAlphabet alphabet = new IntAlphabet(ALPHABET_SIZE);
    AdamDescent adamDescent = new AdamDescent(random, epochCount, 4, gradientStep);
    if (validateEachEpoch) {
      adamDescent.setListener(x -> {
        PNFAModel<Integer> model = new PNFAModel<>(
            x,
            stateCount,
            1,
            addToDiag,
            alpha,
            alphabet,
            bettaParametrization,
            weightParametrization
        );
        System.out.println("Train accuracy: " + (1 - getAccuracy(train, trainTarget, model)));
        System.out.println("Test accuracy: " + (1 - getAccuracy(test, testTarget, model)));
      });
    }
    final GradientSeqBoosting<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
        new BootstrapSeqOptimization<>(
            new PNFARegressor<>(stateCount, stateCount,1, alphabet, alpha, 0.000, addToDiag, 0.1, random,
               // new SAGADescent(0.002, WEIGHTS_EPOCH_COUNT * train.size(), random, System.out),
                adamDescent,
                bettaParametrization, weightParametrization),
            random, 1),
        BOOST_ITERS, boostStep
    );


    Consumer<Function<Seq<Integer>,Vec>> listener = classifier -> {
      try {
        Files.write(Paths.get("kek/1"), new ObjectMapper().writeValueAsString(classifier).getBytes());
      }
      catch (IOException e) {
        e.printStackTrace();
      }

      System.out.println("Current time: " + new SimpleDateFormat("yyyy/MM/dd_HH:mm:ss").format(Calendar.getInstance().getTime()));
      System.out.println("Current accuracy:");
      System.out.println("Train accuracy: " + getAccuracy(train, trainTarget, classifier));
      System.out.println("Test accuracy: " + getAccuracy(test, testTarget, classifier));
      System.out.println("Train loss: " + getLoss(train, trainTarget, classifier));
      System.out.println("Test loss: " + getLoss(test, testTarget, classifier));
    };

    boosting.addListener(listener);
    final Function<Seq<Integer>, Vec> classifier = boosting.fit(data, new LLLogit(trainTarget, null));

    System.out.println("Train accuracy: " + getAccuracy(train, trainTarget, classifier));
    System.out.println("Test accuracy: " +  getAccuracy(test, testTarget, classifier));
  }

  private void readWordData(String path, List<Seq<Integer>> data, Vec target) throws IOException {
    List<String> list = Files.readAllLines(Paths.get(path));
    Collections.shuffle(list, random);

    for (int i = 0; i < TRAIN_SIZE; i++) {
      String[] tokens = list.get(i).split(" ");
      target.set(i, Integer.parseInt(tokens[0]));
      data.add(new IntSeq(Arrays.stream(tokens, 1, tokens.length).mapToInt(Integer::parseInt).toArray()));
    }
  }

  private List<CharSeq> readData(final String filePath) throws IOException {
    long start = System.nanoTime();
    final List<CharSeq> data = Files.list(Paths.get(filePath)).map(path -> {
      try {
        return  new CharSeqArray(Files.readAllLines(path)
            .stream()
            .map(String::toLowerCase)
            .map(str -> str.replaceAll("[^a-zA-Z]", ""))
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

  private double getAccuracy(List<Seq<Integer>> data, Vec target, Function<Seq<Integer>, Vec> classifier) {
    int passedCnt = 0;
    for (int i = 0; i < data.size(); i++) {
      final double val = classifier.apply(data.get(i)).get(0);
      if ((target.get(i) > 0 && val > 0) || (target.get(i) <= 0 && val <= 0)) {
        passedCnt++;
      }
    }
    return 1.0 * passedCnt / data.size();
  }

  private double getLoss(List<Seq<Integer>> data, Vec target, Function<Seq<Integer>, Vec> classifier) {
    final LLLogit lllogit = new LLLogit(target, null);
    Vec values = new ArrayVec(target.dim());
    for (int i =0 ; i < target.dim(); i++) {
      values.set(i, classifier.apply(data.get(i)).get(0));
    }
    return lllogit.value(values);
  }

  private static Options options = new Options();
  static {
    options.addOption("stateCount", true, "stateCount");
    options.addOption("alpha", true, "alpha");
    options.addOption("addToDiag", true, "addToDiag");
    options.addOption("boostStep", true, "boostStep");
    options.addOption("epochs", true, "epochs");
    options.addOption("step", true, "step");
    options.addOption("wParam", true, "wParam");
    options.addOption("bParam", true, "bParam");
    options.addOption("validateEpoch", true, "validateEpoch");
    options.addOption("alphabet", true, "alphabet");
  }
  public static void main(String[] args) throws IOException, ParseException {
    final CommandLineParser parser = new GnuParser();
    final CommandLine command = parser.parse(options, args);
    System.out.println(Arrays.toString(args));
    RunImdb test = new RunImdb(
        Integer.parseInt(command.getOptionValue("stateCount")),
        Double.parseDouble(command.getOptionValue("alpha")),
        Double.parseDouble(command.getOptionValue("addToDiag")),
        Double.parseDouble(command.getOptionValue("boostStep")),
        Integer.parseInt(command.getOptionValue("epochs")),
        Double.parseDouble(command.getOptionValue("step")),
        command.getOptionValue("wParam"),
        command.getOptionValue("bParam"),
        Boolean.parseBoolean(command.getOptionValue("validateEpoch"))
        );
    if (command.getOptionValue("alphabet").equals("vgram")) {
      test.loadData();
    } else {
      test.loadWordData();
    }
    test.test();
  }

}
