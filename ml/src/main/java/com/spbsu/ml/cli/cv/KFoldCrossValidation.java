package com.spbsu.ml.cli.cv;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.TargetFunc;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.cli.builders.methods.MethodsBuilder;
import com.spbsu.ml.cli.gridsearch.ParametersGridEnumerator;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.methods.VecOptimization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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
      threadPool.execute(new Runnable() {
        @Override
        public void run() {
          LOG.info("Starting evaluation for parameters: " + Arrays.toString(parameters));
          final double meanScore = evaluate(concreteScheme);
          LOG.info("Mean score = " + meanScore + " for parameters " + Arrays.toString(parameters));
          synchronized (bestParametersHolder) {
            if (bestParametersHolder.update(parameters, meanScore)) {
              LOG.debug("The best score was updated. Score = " + bestParametersHolder.getScore() + ", parameters: " + Arrays.toString(parameters));
            };
          }
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
      LOG.debug("Fold #" + i + ", score = " + score + " for scheme: " + concreteTrainingScheme);
      totalScore += score;
    }

    return totalScore / folds.size();
  }
}
