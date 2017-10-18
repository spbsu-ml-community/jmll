package com.expleague.ml.cli.cv;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.logging.Logger;
import com.expleague.commons.math.stat.WXTest;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.BestHolder;
import com.expleague.commons.util.Pair;
import com.expleague.ml.TargetFunc;
import com.expleague.commons.math.Trans;
import com.expleague.ml.cli.builders.methods.MethodsBuilder;
import com.expleague.ml.cli.gridsearch.ParametersGridEnumerator;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.methods.VecOptimization;
import gnu.trove.list.array.TDoubleArrayList;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class KFoldCrossValidation {
  private static final Logger LOG = Logger.create(KFoldCrossValidation.class);

  private static final int THREADS_COUNT = Runtime.getRuntime().availableProcessors();

  private final String targetClassName;
  private final MethodsBuilder methodsBuilder;

  private final List<Pair<? extends Pool, ? extends Pool>> folds;

  public KFoldCrossValidation(
      final Pool sourcePool,
      final FastRandom random,
      final int foldsCount,
      final String targetClassName,
      final MethodsBuilder methodsBuilder
  ) {
    this.targetClassName = targetClassName;
    this.methodsBuilder = methodsBuilder;

    this.folds = new ArrayList<>();
    final FoldsEnumerator foldsEnumerator = new FoldsEnumerator(sourcePool, random, foldsCount);
    while (foldsEnumerator.hasNext()) {
      folds.add(foldsEnumerator.next());
    }
  }

  public BestHolder<Object[]> evaluateParametersRange(final String commonScheme, final Object[][] parametersSpace) {
    final ExecutorService threadPool = Executors.newFixedThreadPool(THREADS_COUNT);

    final BestHolder<Object[]> bestParametersHolder = new BestHolder<>();
    final ParametersGridEnumerator<?> parametersEnumerator = new ParametersGridEnumerator<>(parametersSpace);
    while (parametersEnumerator.advance()) {
      final Object[] parameters = parametersEnumerator.getParameters();
      final String concreteScheme = String.format(commonScheme, parameters);
      threadPool.execute(() -> {
        LOG.info("Starting evaluation for parameters: " + Arrays.toString(parameters));
        final double meanScore = evaluate(concreteScheme);
        LOG.info("Mean score = " + meanScore + " for parameters " + Arrays.toString(parameters));
        synchronized (bestParametersHolder) {
          if (bestParametersHolder.update(parameters, meanScore)) {
            LOG.debug("The best score was updated. Score = " + bestParametersHolder.getScore() + ", parameters: " + Arrays.toString(parameters));
          };
        }
      });
    }
    try {
      threadPool.shutdown();
      threadPool.awaitTermination(7, TimeUnit.DAYS);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return bestParametersHolder;
  }

  public double evaluate(final String concreteTrainingScheme) {
    final VecOptimization method = methodsBuilder.create(concreteTrainingScheme);
    double totalScore = 0.;
    for (int i = 0; i < folds.size(); i++) {
      final Pair<? extends Pool, ? extends Pool> learnAndTest = folds.get(i);
      final Pool learnPool = learnAndTest.getFirst();
      final Pool testPool = learnAndTest.getSecond();

      final TargetFunc learnLoss = learnPool.target(DataTools.targetByName(targetClassName));
      final TargetFunc testLoss = testPool.target(DataTools.targetByName(targetClassName));

      final Trans model = method.fit(learnPool.vecData(), learnLoss);
      final Vec predict = DataTools.calcAll(model, testPool.vecData());
      final double score = testLoss.value(predict);
      LOG.debug("Fold #" + i + ", score = " + score + " for schemes: " + concreteTrainingScheme);
      totalScore += score;
    }

    return totalScore / folds.size();
  }


  public static class CrossValidationModelComparisonResult {
    final List<String> schemes;
    final Mx WxStats;
    final Mx pairwiseDiffs;
    final Vec scores;

    CrossValidationModelComparisonResult(final List<String> schemes,
                                         final Mx WxStats,
                                         final Mx pairwiseDiffs,
                                         final Vec scores) {
      this.schemes = schemes;
      this.WxStats = WxStats;
      this.pairwiseDiffs = pairwiseDiffs;
      this.scores = scores;
    }


    public Stream<String> getSchemes() {
      return schemes.stream();
    }

    public Mx getWxStats() {
      return WxStats;
    }

    public Mx getPairwiseDiffs() {
      return pairwiseDiffs;
    }

    public Vec getScores() {
      return scores;
    }

    static class ModelComparisionResultBuilder {
      final Map<String, TDoubleArrayList> foldScores;
      int currentFold = -1;

      ModelComparisionResultBuilder(final List<String> schemeNames) {
        foldScores = new HashMap<>();
        for (String scheme : schemeNames) {
          foldScores.put(scheme, new TDoubleArrayList());
        }
      }

      ModelComparisionResultBuilder startNextFold() {
        ++currentFold;
        return this;

      }

      ModelComparisionResultBuilder addScore(final String model,
                                             double score) {
        final TDoubleArrayList schemeScores = foldScores.get(model);
        if (schemeScores.size() != currentFold) {
          throw new RuntimeException("error: folds should be consistent");
        }
        schemeScores.add(score);
        return this;
      }

      CrossValidationModelComparisonResult build() {

        final List<String> schemes = new ArrayList<>(foldScores.keySet());
        final Mx scoreDiffs = new VecBasedMx(schemes.size(), schemes.size());
        final Vec scores = new ArrayVec(schemes.size());
        final Mx pValues = new VecBasedMx(schemes.size(), schemes.size());

        for (int i = 0; i < schemes.size(); ++i) {
          final String firstScheme = schemes.get(i);
          scores.set(i, calcScore(foldScores.get(firstScheme)));
          for (int j = 0; j  < i; ++j) {
            final String secondScheme = schemes.get(j);
            final double wx = WXTest.test(foldScores.get(firstScheme), foldScores.get(secondScheme));
            final double stat = wx < 0.5 ? (1.0 - wx) : wx;
            pValues.set(i, j, stat);
            pValues.set(j, i, stat);
          }
        }

        for (int i = 0; i < schemes.size(); ++i) {
          for (int j = 0; j  < schemes.size(); ++j) {
            scoreDiffs.set(i, j, scores.get(i) - scores.get(j));
          }
        }
        return new CrossValidationModelComparisonResult(schemes, pValues, scoreDiffs, scores);
      }

      private double calcScore(TDoubleArrayList data) {
        double score = 0;
        for (int i = 0; i < data.size(); ++i) {
          score += data.get(i);
        }
        return score / data.size();
      }
    }
  }

  public CrossValidationModelComparisonResult evaluateSchemesBatch(final List<String> schemes) {

    final CrossValidationModelComparisonResult.ModelComparisionResultBuilder cvBuilder = new CrossValidationModelComparisonResult.ModelComparisionResultBuilder(schemes);

    for (int i = 0; i < folds.size(); i++) {
      cvBuilder.startNextFold();

      for (String optimizationScheme : schemes) {
        final VecOptimization method = methodsBuilder.create(optimizationScheme);
        final Pair<? extends Pool, ? extends Pool> learnAndTest = folds.get(i);
        final Pool learnPool = learnAndTest.getFirst();
        final Pool testPool = learnAndTest.getSecond();

        final TargetFunc learnLoss = learnPool.target(DataTools.targetByName(targetClassName));
        final TargetFunc testLoss = testPool.target(DataTools.targetByName(targetClassName));

        final Trans model = method.fit(learnPool.vecData(), learnLoss);
        final Vec predict = DataTools.calcAll(model, testPool.vecData());
        final double score = testLoss.value(predict);
        cvBuilder.addScore(optimizationScheme, score);
        LOG.info("Fold #" + i + ", score = " + score + " for schemes: " + optimizationScheme);
      }
    }
    return cvBuilder.build();
  }
}
