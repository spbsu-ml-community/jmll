package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.stat.StatTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

import static com.spbsu.commons.math.MathTools.sqr;

/**
 * Created by noxoomo on 10/06/15.
 */
public class WeakLeastAngle implements VecOptimization<L2> {

  public static class WeakLinear extends Func.Stub {
    final int dim;
    final int condition;
    final double value;

    public WeakLinear(int dim, int condition, double value) {
      this.dim = dim;
      this.condition = condition;
      this.value = value;
    }

    @Override
    public double value(Vec x) {
      return x.get(condition) * value;
    }

    @Override
    public int dim() {
      return dim;
    }
  }

  final private int[] points;
  final private int[] features;

  public WeakLeastAngle(int[] points, int[] features) {
    this.points = points;
    this.features = features;
  }

  public WeakLeastAngle() {
    this.points = null;
    this.features = null;
  }


  @Override
  public WeakLinear fit(VecDataSet learn, L2 l2) {
    final int[] points;
    final int[] features;
    final Mx data = learn.data();
    final Vec target = l2.target();

    if (this.points == null) {
      points = ArrayTools.sequence(0,data.rows());
      features = ArrayTools.sequence(0,data.columns());
    } else {
      points = this.points;
      features = this.features;
    }


    double variance = StatTools.variance(target);

    double bestInc = 0;
    double leastAngle = 0;
    int bestInd = 0;
    final double targetNorm = Math.sqrt(multiply(target, target, points));

    for (int i : features) {
      final Vec feature = data.col(i);
      final double featureNorm =Math.sqrt(multiply(feature,feature,points));
      final double dotProd = multiply(feature, target, points);
      double angle = dotProd / featureNorm / targetNorm;
      if (Math.abs(angle) > leastAngle) {
        leastAngle = Math.abs(angle);
        bestInd = i;
        bestInc = dotProd / featureNorm / featureNorm;
      }
    }

    final WeakLinear result = new WeakLinear(data.columns(), bestInd, bestInc);
    if (score(result, data, target,points) < variance) {
      return result;
    } else {
      return new WeakLinear(data.columns(), 0, 0);
    }
  }

  private double score(WeakLinear model, Mx data, Vec target, int[] points) {
    double score = 0;
    for (int i : points) {
      final double diff = sqr(model.value(data.row(i)) - target.get(i));
      score += diff;
    }
    return score / (data.rows() - 2);
  }

  private double multiply(Vec left, Vec right, int[] points) {
    double res = 0;
    for (int i : points) {
      res += left.get(i) * right.get(i);
    }
    return res;
  }


}
