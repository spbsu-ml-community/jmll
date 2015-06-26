package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.stat.StatTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import gnu.trove.list.array.TIntArrayList;

import static com.spbsu.commons.math.MathTools.sqr;

/**
 * Created by noxoomo on 10/06/15.
 */
public class ValidationRidgeRegression implements VecOptimization<L2> {
  final double validation;
  final FastRandom rand;
  final private double minLambda = 1e-1;

  public ValidationRidgeRegression(double validationPart, FastRandom rand) {
    this.validation = validationPart;
    this.rand = rand;
  }

  @Override
  public Linear fit(VecDataSet learn, L2 l2) {
    FastRandom random = new FastRandom(rand.nextLong()); //for parallel fit

    Mx data = learn.data();
    TIntArrayList learnPoints = new TIntArrayList();
    TIntArrayList validationPoints = new TIntArrayList();
    for (int i = 0; i < data.rows(); ++i) {
      if (random.nextDouble() < validation) {
        validationPoints.add(i);
      } else {
        learnPoints.add(i);
      }
    }

    Vec target = l2.target();
    double variance = StatTools.variance(target);

    Mx cov = new VecBasedMx(data.columns(), data.columns());
    Vec covTargetWithFeatures = new ArrayVec(data.columns());


    for (int i = 0; i < data.columns(); ++i) {
      final Vec feature = data.col(i);
      cov.set(i, i, multiply(feature, feature, learnPoints));
      covTargetWithFeatures.set(i, multiply(feature, target, learnPoints));
      for (int j = i + 1; j < data.columns(); ++j) {
        final double val = multiply(feature, data.col(j), learnPoints);
        cov.set(i, j, val);
        cov.set(j, i, val);
      }
    }

    RidgeRegressionCache ridge = new RidgeRegressionCache(cov, covTargetWithFeatures);
    double bestScore = variance;
    double lambda = minLambda;
    double bestLambda = lambda;
    while (true) {
      final Linear model = ridge.fit(lambda);
      final double score = score(model, data, target, validationPoints);
      if (score > bestScore) {
        break;
      }
      bestLambda = lambda;
      lambda *= 2;
      bestScore = score;
      if (lambda > 1) {
        return new Linear(new double[data.columns()]);
      }
    }
    if (bestScore <= variance) {
      return new Linear(new double[data.columns()]);
    }

    for (int i = 0; i < data.columns(); ++i) {
      final Vec feature = data.col(i);
      cov.adjust(i, i, multiply(feature, feature, validationPoints));
      covTargetWithFeatures.adjust(i, multiply(feature, target, validationPoints));
      for (int j = i + 1; j < data.columns(); ++j) {
        final double val = multiply(feature, data.col(j), validationPoints);
        cov.adjust(i, j, val);
        cov.adjust(j, i, val);
      }
    }

    Linear result = ridge.fit(bestLambda);
    learnPoints.addAll(validationPoints);
    double resultScore = score(result, data, target, learnPoints);
    if (resultScore > variance) {
      return new Linear(new double[data.columns()]);
    }
    return result;
  }

  private double score(Linear model, Mx data, Vec target, TIntArrayList points) {
    double score = 0;
    for (int i = 0; i < points.size(); ++i) {
      final int point = points.get(i);
      final double diff = sqr(model.value(data.row(point)) - target.get(point));
      score += diff;
    }
    return score / (points.size() - model.dim());
  }

  private double multiply(Vec left, Vec right, TIntArrayList points) {
    double res = 0;
    for (int i = 0; i < points.size(); ++i) {
      final int ind = points.get(i);
      res += left.get(ind) * right.get(ind);
    }
    return res;
  }
}
