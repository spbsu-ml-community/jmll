package com.expleague.basecalling;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.Dictionary;
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
import com.expleague.ml.methods.seq.BootstrapSeqOptimization;
import com.expleague.ml.methods.seq.GradientSeqBoosting;
import com.expleague.ml.methods.seq.PNFA;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;
import com.expleague.ml.optimization.impl.FullGradientDescent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class PNFABasecall {
  private static final int WEIGHT_EPOCH_COUNT = 30;
  private static final int VALUE_EPOCH_COUNT = 10;
  private static final int BATCH_SIZE = 16;
  private static final int BOOST_ITERS = 300;

  private static final double WEIGHT_STEP = 0.0003;
  private static final double VALUE_STEP = 1;
  private static final int ALPHABET_SIZE = 2500;
  private final int alphabetSize;

  private final static String NUCLEOTIDES = "ACGT";
  private final static int CLASS_COUNT = 4;

  private final DataSet<Seq<Integer>> trainDataSet;
  private final DataSet<Seq<Integer>> testDataSet;
  private final IntSeq trainLabels;
  private final IntSeq testLabels;
  private final FastRandom random;

  private final double boostStep;
  private final double lambda;
  private final double addToDiag;
  private final int stateCount;

  public PNFABasecall(final Path datasetPath,
                      final int stateCount,
                      final double lambda,
                      final double addToDiag,
                      final double boostStep,
                      final double trainPart,
                      final double testPart,
                      final FastRandom random,
                      final boolean useDifferences) throws IOException {
    this.random = random;
    this.stateCount = stateCount;
    this.lambda = lambda;
    this.addToDiag = addToDiag;
    this.boostStep = boostStep;

    final List<IntSeq> trainOld = new ArrayList<>();
    final List<IntSeq> testOld = new ArrayList<>();
    final List<Integer> trainClass = new ArrayList<>();
    final List<Integer> testClass = new ArrayList<>();

    Files.readAllLines(datasetPath).forEach(line -> {
      final String[] tokens = line.split(" ");
      final int clazz = NUCLEOTIDES.indexOf(tokens[0]);
      final int[] signal = Arrays
          .stream(tokens[1].split(","))
          .mapToInt(Integer::parseInt)
          .toArray();
      final IntSeq seq = useDifferences ? getDiffSeq(signal) : new IntSeq(signal);


      final double rnd = random.nextDouble();
      if (rnd < trainPart) {
        trainOld.add(seq);
        trainClass.add(clazz);
      } else if (rnd < trainPart + testPart) {
        testOld.add(seq);
        testClass.add(clazz);
      }
    });

    final int minLevel = trainOld.stream().flatMapToInt(seq -> ((IntSeq) seq).stream()).min()
        .getAsInt();
    final int maxLevel = trainOld.stream().flatMapToInt(seq -> ((IntSeq) seq).stream()).max().getAsInt();
    //    final int minLevel = -4098;
    //    final int maxLevel = 4097;
    List<IntSeq> all = new ArrayList<>(trainOld);
    all.addAll(testOld);
    DictExpansion<Integer> de = new DictExpansion<>(all.stream().flatMapToInt(it -> Arrays
        .stream(it.toArray()))
        .sorted()
        .distinct()
        .boxed()
        .collect(Collectors.toList()), ALPHABET_SIZE, System.out);
    System.out.println("Original dict size: " + de.alpha().size());
    for (int i = 0; i < 4; i++) {
      all.forEach(de::accept);
    }

    Dictionary<Integer> dictionary = de.result();

    final List<IntSeq> train = trainOld.stream().map(dictionary::parse).collect(Collectors.toList());
    final List<IntSeq> test = testOld.stream().map(dictionary::parse).collect(Collectors.toList());
    System.out.println("Min level: " + minLevel + ", max level: " + maxLevel);
    alphabetSize = dictionary.alphabet().size();
    System.out.println("alphabetSize= " + alphabetSize);

    trainDataSet = new DataSet.Stub<Seq<Integer>>(null) {
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
    testDataSet = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return test.get(i);
      }

      @Override
      public int length() {
        return test.size();
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    };

    trainLabels = new IntSeq(trainClass.stream().mapToInt(Integer::intValue).toArray());
    testLabels = new IntSeq(testClass.stream().mapToInt(Integer::intValue).toArray());

    System.out.println("Train size: " + train.size());
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
    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new AdamDescent(
        random, WEIGHT_EPOCH_COUNT, BATCH_SIZE, WEIGHT_STEP
    );
    final Optimize<FuncEnsemble<? extends FuncC1>> valueOptimizer  = new FullGradientDescent(
        random, VALUE_STEP, VALUE_EPOCH_COUNT
    );
    final PNFA model = new PNFA<>(stateCount,
        1,
        alphabetSize,
        lambda,
        addToDiag,
        random,
        weightOptimizer,
        valueOptimizer,
        2
    );
    final GradFacMulticlassSeq<Integer> multiClassModel = new GradFacMulticlassSeq<Integer>(
        model,
        new StochasticALS(random,100),
        WeightedL2.class //todo -> weighted log l2 ?
    );

    fitBoosting(multiClassModel, globalLoss, true);
  }

  private <Loss extends TargetFunc> void fitBoosting(
      SeqOptimization<Integer, L2> model,
      Loss loss,
      boolean negativeIsLastClass
  ) {
    final GradientSeqBoosting<Integer, Loss> boosting = new GradientSeqBoosting<>(
        model, BOOST_ITERS, boostStep
    );
    Consumer<Function<Seq<Integer>, Vec>> listener = it -> printProgress(it, negativeIsLastClass);
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
  //        new PNFA<>(stateCount,
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
      l2LossVec.set(i * CLASS_COUNT + trainLabels.at(i), 1);
    }
    final L2 globalLoss = new L2(l2LossVec, trainDataSet);
    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new AdamDescent(
        random, WEIGHT_EPOCH_COUNT, BATCH_SIZE, WEIGHT_STEP
    );
    final Optimize<FuncEnsemble<? extends FuncC1>> valueOptimizer  = new FullGradientDescent(
        random, VALUE_STEP, VALUE_EPOCH_COUNT
    );
    final SeqOptimization<Integer, L2> model = new BootstrapSeqOptimization<>(
        new PNFA<>(stateCount,
            CLASS_COUNT,
            alphabetSize,
            lambda,
            addToDiag,
            random,
            weightOptimizer,
            valueOptimizer,
            2
        ),
        random
    );
    final GradientSeqBoosting<Integer, L2> boosting = new GradientSeqBoosting<>(
        model, BOOST_ITERS, boostStep
    );

    fitBoosting(model, globalLoss, false);
  }

  private void printProgress(Function<Seq<Integer>, Vec> model, boolean negativeIsLastClass) {
    final IntSeq trainPred = predict(model, trainDataSet, negativeIsLastClass);
    final IntSeq testPred = predict(model, testDataSet, negativeIsLastClass);
    System.out.println(testPred);
    System.out.println("Train confusion: " + new ConfusionMatrix(trainLabels, trainPred).oneLineReport());
    System.out.println("Test confusion: " + new ConfusionMatrix(testLabels, testPred).oneLineReport());
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
}
