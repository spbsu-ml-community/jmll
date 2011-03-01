package ml.models;

import ml.data.DataEntry;

/**
* User: solar
* Date: 01.03.11
* Time: 22:30
*/
public class NormalizedLinearModel extends LinearModel {
  private final double[] means;
  private final double[] norms;
  private final double meanTarget;

  public NormalizedLinearModel(double[] betas, double means[], double[] norms, double learnScore, double meanTarget) {
    super(betas, learnScore);
    this.means = means;
    this.norms = norms;
    this.meanTarget = meanTarget;
  }

  public double value(DataEntry point) {
      double result = 0;
      for (int i = 0; i < betas.length; i++) {
          if (norms[i] == 0) continue;
          result += (point.x(i) - means[i]) / norms[i] * betas[i];
      }
      return result + meanTarget;
  }
}
