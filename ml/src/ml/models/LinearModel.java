package ml.models;

import ml.Model;
import ml.data.DataEntry;

import java.util.Arrays;
import java.util.Comparator;

/**
* User: solar
* Date: 01.03.11
* Time: 22:30
*/
public class LinearModel implements Model {
  protected final double[] betas;
  private final double learnScore;

  public LinearModel(double[] betas, double learnScore) {
    this.betas = betas;
    this.learnScore = learnScore;
  }

  public double value(DataEntry point) {
      double result = 0;
      for (int i = 0; i < betas.length; i++) {
          result += point.x(i) * betas[i];
      }
      return result;
  }

  public double learnScore() {
      return learnScore;
  }

  public String toString() {
      String result = "";
      Integer[] order = new Integer[betas.length];
      for (int k = 0; k < betas.length; k++) {
          order[k] = k;
      }
      Arrays.sort(order, new Comparator<Integer>() {
        public int compare(Integer a, Integer b) {
          return (int) Math.signum(Math.abs(betas[b]) - Math.abs(betas[a]));
        }
      });
      for (int i = 0; i < betas.length; i++) {
          if(betas[order[i]] == 0)
              continue;
          result += "\t" + order[i] + ": " + betas[order[i]] + "\n";
      }
      return result;
  }

  public final double beta(int i) {
    return betas[i];
  }
}
