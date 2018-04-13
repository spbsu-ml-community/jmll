package com.expleague.basecalling;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multiclass.util.ConfusionMatrix;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlassSeq;
import com.expleague.ml.methods.seq.*;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;
import com.expleague.ml.optimization.impl.FullGradientDescent;
import com.expleague.ml.optimization.impl.OnlineDescent;
import com.expleague.ml.optimization.impl.SAGADescent;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;

public class PNFABasecall {
  private static final int WEIGHT_EPOCH_COUNT = 10;
  private static final int VALUE_EPOCH_COUNT = 3;
  private static final int BATCH_SIZE = 16;
  private static final int BOOST_ITERS = 30;

  private static final double WEIGHT_STEP = 0.0003;
  private static final double VALUE_STEP = 1;
  private static final int ALPHABET_SIZE = 1000;
  private static final int WEIGHT_VALUE_ITERS = 2;

  private final static String NUCLEOTIDES = "ACGT";
  private final static int CLASS_COUNT = 4;

  private final DataSet<Seq<Integer>> trainDataSet;
  private final DataSet<Seq<Integer>> testDataSet;
  private final IntSeq trainLabels;
  private final IntSeq testLabels;
  private final int randomSeed;
  private final FastRandom random;

  private final double boostStep;
  private final double alpha;
  private final double addToDiag;
  private final int stateCount;
  private final Set<Integer> alphabet;
  private final Path checkpointPath;
  private final Path datasetPath;
  private final boolean useDifferences;
  private final int alphaShrink;

  public PNFABasecall(final Path datasetPath,
                      final Path checkpointPath,
                      final int stateCount,
                      final int alphaShrink,
                      final double alpha,
                      final double addToDiag,
                      final double boostStep,
                      final double trainPart,
                      final double testPart,
                      final int randomSeed,
                      final boolean useDifferences) throws IOException {
    this.datasetPath = datasetPath;
    this.checkpointPath = checkpointPath;
    this.randomSeed = randomSeed;
    this.random = new FastRandom(randomSeed);
    this.stateCount = stateCount;
    this.alphaShrink = alphaShrink;
    this.alpha = alpha;
    this.addToDiag = addToDiag;
    this.boostStep = boostStep;
    this.useDifferences = useDifferences;

    final List<IntSeq> train = new ArrayList<>();
    final List<IntSeq> test = new ArrayList<>();
    final List<Integer> trainClass = new ArrayList<>();
    final List<Integer> testClass = new ArrayList<>();

    Files.readAllLines(datasetPath).forEach(line -> {
      final String[] tokens = line.split(" ");
      final int clazz = NUCLEOTIDES.indexOf(tokens[0]);
      final int[] signal = Arrays
          .stream(tokens[1].split(","))
          .mapToInt(s -> Integer.parseInt(s) / alphaShrink)
          .toArray();
      final IntSeq seq = useDifferences ? getDiffSeq(signal) : new IntSeq(signal);


      final double rnd = random.nextDouble();
      if (rnd < trainPart) {
        train.add(seq);
        trainClass.add(clazz);
      } else if (rnd < trainPart + testPart) {
        test.add(seq);
        testClass.add(clazz);
      }
    });

    alphabet = getAlphabet(train);
    alphabet.addAll(getAlphabet(test));

    trainLabels = new IntSeq(trainClass.stream().mapToInt(Integer::intValue).toArray());
    testLabels = new IntSeq(testClass.stream().mapToInt(Integer::intValue).toArray());

    trainDataSet = createDataSet(train);
    testDataSet = createDataSet(test);

    System.out.println("Train size: " + train.size());
  }

  private Set<Integer> getAlphabet(List<IntSeq> data) {
    Set<Integer> set = new HashSet<>();
    for (IntSeq seq: data) {
      seq.forEach(set::add);
    }
    return set;
  }

  private IntSeq getDiffSeq(int[] signal) {
    int[] diffSignal = new int[signal.length - 1];
    for (int i = 0; i < signal.length - 1; i++) {
      diffSignal[i] = signal[i + 1] - signal[0];
    }
    return new IntSeq(diffSignal);
  }

  void trainGradFac() {
    final BlockwiseMLLLogit globalLoss = new BlockwiseMLLLogit(trainLabels, trainDataSet);
//    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new AdamDescent(
//        random, WEIGHT_EPOCH_COUNT, BATCH_SIZE, WEIGHT_STEP
//    );
    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new SAGADescent(
        0.1, 100000, random
    );
    IntAlphabet intAlphabet = new IntAlphabet(ALPHABET_SIZE);
    final PNFARegressor model = new PNFARegressor<>(stateCount,
        1, intAlphabet, 0.0001, 0.001,
        0,
        random,
        weightOptimizer);
    final GradFacMulticlassSeq<Integer> multiClassModel = new GradFacMulticlassSeq<Integer>(
        model,
        new StochasticALS(random,100),
        WeightedL2.class //todo -> weighted log l2 ?
    );

    fitBoostingForModel(multiClassModel, globalLoss, true);
  }

  private <Loss extends TargetFunc> void fitBoostingForModel(
      SeqOptimization<Integer, L2> model,
      Loss loss,
      boolean negativeIsLastClass
  ) {
    final DictExpansionOptimization<Integer, L2> optimization = new DictExpansionOptimization<>(
        model, alphabet.size(), alphabet, System.err
    );
    final GradientSeqBoosting<Integer, Loss> boosting = new GradientSeqBoosting<>(
        optimization, BOOST_ITERS, boostStep
    );
    int[] modelCount = new int[1];

    Consumer<Function<Seq<Integer>, Vec>> listener = boostModel -> {
      printProgress(boostModel, negativeIsLastClass);
      modelCount[0]++;
      if (modelCount[0] % 1 == 0) {
        try {
          saveModel(Integer.toString(modelCount[0]), boostModel);
        }
        catch (IOException e) {
          e.printStackTrace();
        }
      }

    };
    boosting.addListener(listener);

    boosting.fit(trainDataSet, loss);

  }

  //  void trainOneVsRest() {
  //    final MCMicroPrecision globalLoss = new MCMicroPrecision(trainLabels, trainDataSet);
  //    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new AdamDescent(
  //        random, WEIGHT_EPOCH_COUNT, BATCH_SIZE, WEIGHT_STEP
  //    );
  //    final Optimize<FuncEnsemble<? extends FuncC1>> valueOptimizer  = new FullGradientDescent(
  //        random, VALUE_STEP, VALUE_EPOCH_COUNT
  //    );
  //    final SeqOptimization<Integer, L2> model = new BootstrapSeqOptimization<>(
  //        new PNFARegressor<>(stateCount,
  //            1,
  //            alphabetSize,
  //            lambda,
  //            addToDiag,
  //            random,
  //            weightOptimizer,
  //            valueOptimizer,
  //            2
  //        ),
  //        random
  //    );
  //    final SeqOptimization<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
  //        model, BOOST_ITERS, boostStep
  //    );
  //
  //    final MultiClassOneVsRestSeq<Integer> multiClassModel = new MultiClassOneVsRestSeq<>(boosting);
  //
  //    final Function<Seq<Integer>, Vec> classifier = multiClassModel.fit(trainDataSet, globalLoss);
  //    printProgress(classifier);
  //  }

  void tranVecPNFA() {
    final Vec l2LossVec = new ArrayVec(trainLabels.length() * CLASS_COUNT);
    for (int i = 0; i < trainLabels.length(); i++) {
      for (int j = 0; j < CLASS_COUNT; j++) {
        l2LossVec.set(i * CLASS_COUNT + j, trainLabels.at(i) == j ? 1 : -1.0 / (CLASS_COUNT - 1));
      }
    }
    final L2 globalLoss = new L2(l2LossVec, trainDataSet);

//    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new OnlineDescent(1e-3, random);
//    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new AdamDescent(
//        random, 50, 4, 0.0001
//    );
    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new SAGADescent(0.001, 1000000, random);
    IntAlphabet alphabet = new IntAlphabet(ALPHABET_SIZE);
    final SeqOptimization<Integer, L2> model = new BootstrapSeqOptimization<>(
      new PNFARegressor<>(
          stateCount,
          CLASS_COUNT, alphabet,
          1e-6, 1e-4,
          10,
          random,
          weightOptimizer), random
    );

    fitBoostingForModel(model, globalLoss, false);
  }

  private void printProgress(Function<Seq<Integer>, Vec> model, boolean negativeIsLastClass) {
    final IntSeq trainPred = predict(model, trainDataSet, negativeIsLastClass);
    final IntSeq testPred = predict(model, testDataSet, negativeIsLastClass);
//    System.out.println(testPred);
    ConfusionMatrix matrix = new ConfusionMatrix(trainLabels, trainPred);
    System.out.println("Train confusion: " + matrix.oneLineReport());
    System.out.println(matrix.toString());

    matrix = new ConfusionMatrix(testLabels, testPred);
    System.out.println("Test confusion: " + matrix.oneLineReport());
    System.out.println(matrix.toString());
  }

  private IntSeq predict(Function<Seq<Integer>, Vec> model, DataSet<Seq<Integer>> dataSet,
                         boolean negativeIsLastClass) {
    int[] pred = new int[dataSet.length()];
    for (int i = 0; i < dataSet.length(); i++) {
      final Vec res = model.apply(dataSet.at(i));
      pred[i] = VecTools.argmax(res);
      if (negativeIsLastClass && res.get(pred[i]) <= 0) {
        pred[i] = res.dim();
      }
    }
    return new IntSeq(pred);
  }

  private DataSet<Seq<Integer>> createDataSet(List<IntSeq> data) {
    return new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return data.get(i);
      }

      @Override
      public int length() {
        return data.size();
      }

      @Override
      public Class<? extends Seq<Integer>> elementType() {
        return null;
      }
    };
  }

  private void saveModel(String modelName, Function<Seq<Integer>, Vec> model) throws IOException {
    final Path savePath = Paths.get(checkpointPath.toString(), modelName);
    if (!checkpointPath.toFile().exists()) {
      checkpointPath.toFile().mkdirs();
    }
    final PNFABasecallModel modelWithMetaData = new PNFABasecallModel(
        datasetPath,
        checkpointPath,
        randomSeed,
        stateCount, alpha,
        addToDiag,
        boostStep,
        useDifferences,
        model
    );

    Files.write(savePath, new ObjectMapper().writeValueAsString(modelWithMetaData).getBytes());
  }

  static class PNFABasecallModel {
    // some metadata
    private Path datasetPath;
    private Path checkpointPath;
    private int randomSeed;
    private int stateCount;
    private double lambda;
    private double addToDiag;
    private double boostStep;
    private boolean useDifferences;

    final GradientSeqBoosting.GradientSeqBoostingModel model;

    PNFABasecallModel(Path datasetPath,
                      Path checkpointPath,
                      int randomSeed,
                      int stateCount,
                      double lambda,
                      double addToDiag,
                      double boostStep,
                      boolean useDifferences,
                      Function<Seq<Integer>, Vec> model) {
      this.datasetPath = datasetPath;
      this.checkpointPath = checkpointPath;
      this.randomSeed = randomSeed;
      this.stateCount = stateCount;
      this.lambda = lambda;
      this.addToDiag = addToDiag;
      this.boostStep = boostStep;
      this.useDifferences = useDifferences;
      this.model = (GradientSeqBoosting.GradientSeqBoostingModel) model;
    }

    public Path getDatasetPath() {
      return datasetPath;
    }

    public void setDatasetPath(Path datasetPath) {
      this.datasetPath = datasetPath;
    }

    public Path getCheckpointPath() {
      return checkpointPath;
    }

    public void setCheckpointPath(Path checkpointPath) {
      this.checkpointPath = checkpointPath;
    }

    public int getRandomSeed() {
      return randomSeed;
    }

    public void setRandomSeed(int randomSeed) {
      this.randomSeed = randomSeed;
    }

    public int getStateCount() {
      return stateCount;
    }

    public void setStateCount(int stateCount) {
      this.stateCount = stateCount;
    }

    public double getLambda() {
      return lambda;
    }

    public void setLambda(double lambda) {
      this.lambda = lambda;
    }

    public double getAddToDiag() {
      return addToDiag;
    }

    public void setAddToDiag(double addToDiag) {
      this.addToDiag = addToDiag;
    }

    public double getBoostStep() {
      return boostStep;
    }

    public void setBoostStep(double boostStep) {
      this.boostStep = boostStep;
    }

    public boolean isUseDifferences() {
      return useDifferences;
    }

    public void setUseDifferences(boolean useDifferences) {
      this.useDifferences = useDifferences;
    }

    public Function<Seq<Integer>, Vec> getModel() {
      return model;
    }
  }
}