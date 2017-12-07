package com.expleague.basecalling;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multiclass.MCMicroPrecision;
import com.expleague.ml.loss.multiclass.util.ConfusionMatrix;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.multiclass.MultiClassOneVsRestSeq;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlassSeq;
import com.expleague.ml.methods.seq.BootstrapSeqOptimization;
import com.expleague.ml.methods.seq.GradientSeqBoosting;
import com.expleague.ml.methods.seq.PNFA;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public class PNFABasecall {
  private static final int STATE_COUNT = 4;
  private static final int WEIGHT_EPOCH_COUNT = 20;
  private static final int VALUE_EPOCH_COUNT = 0;
  private static final int BATCH_SIZE = 4;
  private static final int BOOST_ITERS = 5;
  private static final double BOOST_STEP = 0.01;

  private static final double WEIGHT_STEP = 0.003;
  private static final double VALUE_STEP = 0.001;
  private final int alphabetSize;

  private final static String NUCLEOTIDES = "ACGT";
  private final static int CLASS_COUNT = 4;

  private final DataSet<Seq<Integer>> trainDataSet;
  private final DataSet<Seq<Integer>> testDataSet;
  private final IntSeq trainLabels;
  private final IntSeq testLabels;
  private final FastRandom random;

  public PNFABasecall(final Path datasetPath,
                      final double trainPart,
                      final double testPart,
                      final FastRandom random,
                      final boolean useDifferences) throws IOException {
    this.random = random;

    final List<IntSeq> train = new ArrayList<>();
    final List<IntSeq> test = new ArrayList<>();
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
        train.add(seq);
        trainClass.add(clazz);
      } else if (rnd < trainPart + testPart) {
        test.add(seq);
        testClass.add(clazz);
      }
    });

//    final int minLevel = train.stream().flatMapToInt(seq -> ((IntSeq) seq).stream()).min().getAsInt();
//    final int maxLevel = train.stream().flatMapToInt(seq -> ((IntSeq) seq).stream()).max().getAsInt();
    final int minLevel = -4098;
    final int maxLevel = 4097;

    train.forEach(seq -> {
      for (int i = 0; i < seq.length(); i++) {
        seq.arr[i] -= minLevel;
      }
    });
    test.forEach(seq -> {
      for (int i = 0; i < seq.length(); i++) {
        seq.arr[i] -= minLevel;
      }
    });
    System.out.println("Min level: " + minLevel + ", max level: " + maxLevel);
    alphabetSize = maxLevel - minLevel + 1;

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
    final Optimize<FuncEnsemble<? extends FuncC1>> valueOptimizer  = new AdamDescent(
        random, VALUE_EPOCH_COUNT, BATCH_SIZE, VALUE_STEP
    );
    final PNFA model = new PNFA<>(STATE_COUNT, alphabetSize, random, weightOptimizer,
        valueOptimizer, 2);
    final GradFacMulticlassSeq<Integer> multiClassModel = new GradFacMulticlassSeq<Integer>(
        model,
        new StochasticALS(random,100),
        WeightedL2.class //todo -> weighted log l2 ?
    );

    final GradientSeqBoosting<Integer, BlockwiseMLLLogit> boosting = new GradientSeqBoosting<>(multiClassModel, BOOST_ITERS, BOOST_STEP);
    Consumer<Function<Seq<Integer>, Vec>> listener = this::printProgress;
    boosting.addListener(listener);
    boosting.fit(trainDataSet, globalLoss);
  }

  void trainOneVsRest() {
    final MCMicroPrecision globalLoss = new MCMicroPrecision(trainLabels, trainDataSet);
    final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimizer = new AdamDescent(
        random, WEIGHT_EPOCH_COUNT, BATCH_SIZE, WEIGHT_STEP
    );
    final Optimize<FuncEnsemble<? extends FuncC1>> valueOptimizer  = new AdamDescent(
        random, VALUE_EPOCH_COUNT, BATCH_SIZE, VALUE_STEP
    );
    final SeqOptimization<Integer, L2> model = new BootstrapSeqOptimization<>(
        new PNFA<>(STATE_COUNT, alphabetSize, random, weightOptimizer, valueOptimizer, 2), random
    );
    final SeqOptimization<Integer, LLLogit> boosting = new GradientSeqBoosting<>(
        model, BOOST_ITERS, BOOST_STEP
    );

    final MultiClassOneVsRestSeq<Integer> multiClassModel = new MultiClassOneVsRestSeq<>(boosting);

    final Function<Seq<Integer>, Vec> classifier = multiClassModel.fit(trainDataSet, globalLoss);
    printProgress(classifier);
  }

  private void printProgress(Function<Seq<Integer>, Vec> model) {
    final IntSeq trainPred = predict(model, trainDataSet);
    final IntSeq testPred = predict(model, testDataSet);
    System.out.println(testPred);
    System.out.println("Train confusion: " + new ConfusionMatrix(trainLabels, trainPred).oneLineReport());
    System.out.println("Test confusion: " + new ConfusionMatrix(testLabels, testPred).oneLineReport());
  }

  private IntSeq predict(Function<Seq<Integer>, Vec> model, DataSet<Seq<Integer>> dataSet) {
    int[] pred = new int[dataSet.length()];
    for (int i = 0; i < dataSet.length(); i++) {
      final Vec res = model.apply(dataSet.at(i));
      pred[i] = VecTools.argmax(res);
      if (res.get(pred[i]) <= 0) {
        pred[i] = res.dim();
      }
    }
    return new IntSeq(pred);
  }
}
