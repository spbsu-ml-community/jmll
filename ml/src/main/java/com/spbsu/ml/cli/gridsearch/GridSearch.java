package com.spbsu.ml.cli.gridsearch;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.cli.builders.methods.MethodsBuilder;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.methods.VecOptimization;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 25.03.15
 */
public class GridSearch {
  private Pool<?> learn;
  private Pool<?> test;
  private MethodsBuilder methodsBuilder;
  private TargetFunc loss;
  private Func[] metrics;

  public GridSearch(
      final Pool<?> learn,
      final Pool<?> test,
      final TargetFunc loss,
      final Func[] metrics,
      final MethodsBuilder methodsBuilder
      ) {
    this.learn = learn;
    this.test = test;
    this.methodsBuilder = methodsBuilder;
    this.loss = loss;
    this.metrics = metrics;
  }

  public OptimumHolder[] search(
      final String commonScheme,
      final Object[][] parametersSpace
  ) {
    final OptimumHolder[] optimumHolders = new OptimumHolder[metrics.length];
    final ParametersGridEnumerator<?> enumerator = new ParametersGridEnumerator<>(parametersSpace);
    while (enumerator.advance()) {
      final Object[] parameters = enumerator.getParameters();
      System.out.println(Arrays.toString(parameters));
      final String concreteScheme = String.format(commonScheme, parameters);
      final VecOptimization method = methodsBuilder.create(concreteScheme);
      final Trans result = method.fit(learn.vecData(), loss);
      final double targetValue = loss.value(DataTools.calcAll(result, learn.vecData()));
      final double[] metricsValues = new double[this.metrics.length];
      final Vec testEvaluation = DataTools.calcAll(result, test.vecData());
      for (int i = 0; i < metricsValues.length; i++) {
        metricsValues[i] = metrics[i].value(testEvaluation);
      }
      final OptimumHolder currentHolder = new OptimumHolder(parameters, metricsValues, targetValue);
      for (int i = 0; i < optimumHolders.length; i++) {
        if (currentHolder.isBetterThan(optimumHolders[i], i)) {
          optimumHolders[i] = currentHolder;
        }
      }
    }
    return optimumHolders;
  }

}
