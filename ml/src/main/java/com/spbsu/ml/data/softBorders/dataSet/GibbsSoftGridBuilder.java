package com.spbsu.ml.data.softBorders.dataSet;

import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.softBorders.GibbsExpWeightedPermutationsWalker;
import com.spbsu.ml.data.softBorders.SimpleMCMCRunner;
import com.spbsu.ml.data.softBorders.estimators.BinsEstimator;
import com.spbsu.ml.data.softBorders.estimators.BorderValuesEstimator;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

public class GibbsSoftGridBuilder {
  private static ThreadPoolExecutor threadPoolExecutor = ThreadTools.createBGExecutor("Gibbs walker thread", -1);

  public static SoftGrid build(final VecDataSet ds,
                               final int binFactor) {

    final List<WeightedFeature> features = WeightedFeature.build(ds);
    final List<int[]> borders = features.stream()
            .parallel()
            .map(feature -> GreedyBordersSearcher.borders(feature, binFactor))
            .collect(Collectors.toList());

    final CountDownLatch latch = new CountDownLatch(features.size());
    final double lambda = 0.01;
    final int burnInPerior = 100;
    final int estimationWindow = 5;
    final int runIterations = estimationWindow * 100 + burnInPerior;

    final double[][] bins = new double[features.size()][];
    final double[][] softBorders = new double[features.size()][];

    for (int i = 0; i < features.size(); ++i) {
      final int idx = i;
      threadPoolExecutor.execute(() -> {
        final WeightedFeature feature = features.get(idx);
        if (feature.size() <= 1) {
          bins[idx] = new double[0];
          softBorders[idx] = new double[0];
          latch.countDown();
//          continue;
          return;
        }

        final GibbsExpWeightedPermutationsWalker walker =
                new GibbsExpWeightedPermutationsWalker(feature.size(), lambda, feature.weights());

        final BinsEstimator binsEstimator = new  BinsEstimator(feature, borders.get(idx));
        final BorderValuesEstimator borderValuesEstimator = new BorderValuesEstimator(feature, borders.get(idx));
        final SimpleMCMCRunner<int[]> mcmcRunner = new SimpleMCMCRunner<>();
        mcmcRunner.setSampler(walker)
                .addEsimator(binsEstimator)
                .addEsimator(borderValuesEstimator)
                .setBurnInIterations(burnInPerior)
                .setEstimationWindow(estimationWindow)
                .setRunIterations(runIterations)
                .run();
        bins[idx] = binsEstimator.softBins();
        softBorders[idx] = borderValuesEstimator.borders();

        latch.countDown();
      });
    }
//
    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    final SoftGrid.Builder builder = new SoftGrid.Builder();
    for (int i = 0; i < features.size(); ++i) {
      builder.addRow(features.get(i), bins[i], softBorders[i]);
    }
    final SoftGrid grid = builder.build();
    return grid;
  }

}
