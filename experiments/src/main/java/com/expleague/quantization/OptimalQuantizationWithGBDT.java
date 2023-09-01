package com.expleague.quantization;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.ScoreCalcer;
import com.expleague.ml.cli.output.ModelWriter;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.PoolByRowsBuilder;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.dynamicGrid.interfaces.DynamicRow;
import com.expleague.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.expleague.ml.dynamicGrid.trees.GreedyObliviousTreeDynamic2;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.WeightedBootstrapOptimization;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class OptimalQuantizationWithGBDT {
  public static final int SAMPLES_COUNT = 200_000;

  record PairItem(CharSeq wordA, CharSeq wordB) implements DSItem {

    @Override
    public String id() {
      return wordA + "+" + wordB;
    }
  }
  public static void main(String[] args) throws IOException {
    final String trainFile = "/Users/solar/data/glove/train_glove.txt";
    final String trueNeighboursFile = "/Users/solar/data/glove/neighbors_glove.txt";
    final Map<CharSeq, Vec> wordVectors = new HashMap<>();
    final List<CharSeq> allWords = new ArrayList<>();
    try (final BufferedReader reader = Files.newBufferedReader(Paths.get(trainFile))) {
      final AtomicInteger idCounter = new AtomicInteger();
      CharSeqTools.lines(reader).forEach(line -> {
        final CharSequence[] parts = CharSeqTools.split(line, " ");
        final Vec vec = new ArrayVec(parts.length);
        for (int i = 0; i < parts.length; i++) {
          vec.set(i, CharSeqTools.parseDouble(parts[i]));
        }
//        final CharSeq id = CharSeq.create(parts[0]);
        final CharSeq id = CharSeq.create("" + idCounter.getAndIncrement());
        wordVectors.put(id, vec);
        allWords.add(id);
      });
    }
//    final int[][] trueNeighbours = new HashMap<>();
//    try (final BufferedReader reader = Files.newBufferedReader(Paths.get(trueNeighboursFile))) {
//      CharSeqTools.lines(reader).forEach(line -> {
//        final CharSequence[] parts = CharSeqTools.split(line, " ");
//        final Vec vec = new ArrayVec(parts.length - 1);
//        for (int i = 1; i < parts.length; i++) {
//          vec.set(i - 1, CharSeqTools.parseDouble(parts[i]));
//        }
//        final CharSeq word = CharSeq.create(parts[0]);
//        wordVectors.put(word, vec);
//        allWords.add(word);
//      });
//    }

    final PoolByRowsBuilder<PairItem> poolBuilder = new PoolByRowsBuilder<>(PairItem.class);
    final FastRandom rng = new FastRandom(100500);
    final Vec firstVec = wordVectors.get(allWords.get(0));
    final int vecDim = firstVec.dim();
    poolBuilder.allocateFakeFeatures(2 * vecDim, FeatureMeta.ValueType.VEC);
    poolBuilder.allocateFakeTarget(FeatureMeta.ValueType.VEC);
    for (int i = 0; i < SAMPLES_COUNT; i++) {
      final int wordAId = rng.nextInt(wordVectors.size());
      final int wordBId = rng.nextInt(wordVectors.size());
      final CharSeq wordA = allWords.get(wordAId);
      final CharSeq wordB = allWords.get(wordBId);
      final Vec wordAVec = wordVectors.get(wordA);
      final Vec wordBVec = wordVectors.get(wordB);
      final PairItem item = new PairItem(wordA, wordB);
      for (int f = 0; f < vecDim; f++) {
        poolBuilder.setFeature(f, wordAVec.get(f));
      }
      for (int f = 0; f < vecDim; f++) {
        poolBuilder.setFeature(vecDim + f, wordBVec.get(f));
      }
      poolBuilder.setTarget(0, VecTools.distanceL2(wordAVec, wordBVec));
      poolBuilder.nextItem(item);
    }
    final Pool<PairItem> pool = poolBuilder.create(PairItem.class);
    final List<Pool<PairItem>> poolSplit = DataTools.splitDataSet(pool, rng, 0.8, 0.2);
    final Pool<PairItem> train = poolSplit.get(0);
    final Pool<PairItem> validate = poolSplit.get(1);
    final GreedyObliviousTreeDynamic2<WeightedLoss<? extends L2>> gbdotDynamic;
    gbdotDynamic = new GreedyObliviousTreeDynamic2<>(train.vecData(), 10);
    final Vec target = (Vec)train.target(0);
    final Vec weights = new ArrayVec(train.size());
    for (int i = 0; i < weights.dim(); i++) {
      weights.set(i, 1. / MathTools.sqr(target.get(i)));
    }
    final GradientBoosting<SatL2> boosting = new GradientBoosting<>(
        new WeightedBootstrapOptimization<>(gbdotDynamic, weights, rng),
        L2Reg.class, 4000, 0.02
    );

    { // train
      final ScoreCalcer learnListener = new ScoreCalcer("\n\tlearn:\t", train.vecData(), train.target(L2.class), true, 10);
      final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class), true, 10);
      boosting.addListener(learnListener);
      boosting.addListener(validateListener);
      final Ensemble<ObliviousTreeDynamicBin> result = boosting.fit(train.vecData(), train.target(SatL2.class));
      final ModelWriter modelWriter = new ModelWriter("/Users/solar/temp/model.txt");
      modelWriter.tryWriteGrid(result);
      modelWriter.tryWriteDynamicGrid(result);
      modelWriter.writeModel(result, train);
    }

    final DynamicGrid grid = gbdotDynamic.grid();
    for (int f = 0; f < grid.rows(); f++) {
      final DynamicRow row = grid.row(f);
      final StringBuilder rowRepresentation = new StringBuilder();
      rowRepresentation.append(f).append("(").append(row.size()).append(")");
      for (int b = 0; b < row.size(); b++) {
        rowRepresentation.append(" ").append(row.bf(b));
      }
      System.out.println(rowRepresentation);
    }
  }
}
