package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import static com.spbsu.commons.math.MathTools.sqr;
import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Created by noxoomo on 23/10/14.
 * multitask averaging — http://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/42935.pdf
 * stein — http://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator
 */
public class MTA {
  private final double[][] tasks;
  private final double[] sum;
  private final double[] sum2;
  private final double[] sigma;

  public MTA(double[][] samples) {
    this.tasks = samples;
    sum = new double[samples.length];
    sum2 = new double[samples.length];
    for (int task = 0; task < tasks.length; ++task) {
      for (int i = 0; i < samples[task].length; ++i) {
        double val = samples[task][i];
        sum[task] += val;
        sum2[task] += val * val;
      }
    }
    sigma = new double[tasks.length];
    for (int i = 0; i < tasks.length; ++i) {
      double n = tasks[i].length;
      sigma[i] = n > 1 ? (sum2[i] - sum[i] * sum[i] / n) / (n - 1) : 0;
      sigma[i] /= n;
    }
  }

  public double[] stein(double[] prior) {
    double[] means = new double[sum.length];
    for (int i = 0; i < means.length; ++i) {
      means[i] = sum[i] / tasks[i].length;
    }
    double[] sigma = new double[sum.length];
    double sigmaMax = Double.NEGATIVE_INFINITY;
    double sigmaSum = 0;
    for (int i = 0; i < sigma.length; ++i) {
      final int n = tasks[i].length;
      sigma[i] = n > 1 ? (sum2[i] - sum[i] * sum[i] / n) / (n - 1) : sum2[i] - sum[i] * sum[i];
      sigmaMax = Math.max(sigmaMax, sigma[i]);
      sigmaSum += sigma[i];
    }
    double lambda = sigmaSum / sigmaMax - 2;
    double denum = 0;
    for (int i = 0; i < sum.length; ++i) {
      double diff = (means[i] - prior[i]);
      denum += diff * diff / sigma[i];
    }
    lambda /= denum;
    lambda = 1 - lambda;
    lambda = lambda > 0 ? lambda : 0;
    for (int i = 0; i < means.length; ++i) {
      means[i] = prior[i] + lambda * (means[i] - prior[i]);
    }
    return means;
  }

  public static Vec naiveStein(Vec means) {
    Vec js = copy(means);
    scale(js, (1 - (js.dim() - 2) / sqr(norm(means))));
    return js;
  }

  public double[] stein() {
    double prior = 0;
    double[] means = new double[sum.length];
    for (int i = 0; i < means.length; ++i) {
      means[i] = sum[i] / tasks[i].length;
      prior += means[i];
    }
    prior /= tasks.length;
    double[] sigma = new double[sum.length];
    double sigmaMax = Double.NEGATIVE_INFINITY;
    double sigmaSum = 0;
    for (int i = 0; i < sigma.length; ++i) {
      final int n = tasks[i].length;
      sigma[i] = n > 1 ? (sum2[i] - sum[i] * sum[i] / n) / (n - 1) : sum2[i] - sum[i] * sum[i];
      sigmaMax = Math.max(sigmaMax, sigma[i]);
      sigmaSum += sigma[i];
    }
    double lambda = sigmaSum / sigmaMax - 3;
    double denum = 0;
    for (int i = 0; i < sum.length; ++i) {
      double diff = (means[i] - prior) * tasks[i].length;
      denum += diff * diff / sigma[i];
    }
    lambda = denum > 0 ? 1 - lambda / denum : 0;
    lambda = lambda > 0 ? lambda : 0;
    for (int i = 0; i < means.length; ++i) {
      means[i] = prior + lambda * (means[i] - prior);
    }
    return means;
  }


  public double[] stein(double prior) {
    double[] means = new double[sum.length];
    for (int i = 0; i < means.length; ++i) {
      means[i] = sum[i] / tasks[i].length;
    }
    if (tasks.length < 4) {
      return means;
    }
    double[] sigma = new double[sum.length];
    double sigmaMax = Double.NEGATIVE_INFINITY;
    double sigmaSum = 0;
    for (int i = 0; i < sigma.length; ++i) {
      final int n = tasks[i].length;
      sigma[i] = n > 1 ? (sum2[i] - sum[i] * sum[i] / n) / (n - 1) : sum2[i] - sum[i] * sum[i];
      sigmaMax = Math.max(sigmaMax, sigma[i]);
      sigmaSum += sigma[i];
    }
    double lambda = sigmaSum / sigmaMax - 3;
    double denum = 0;
    for (int i = 0; i < sum.length; ++i) {
      double diff = (means[i] - prior) * tasks[i].length;
      denum += diff * diff / sigma[i];
    }
    lambda = denum > 0 ? 1 - lambda / denum : 0;
    lambda = lambda > 0 ? lambda : 0;
    for (int i = 0; i < means.length; ++i) {
      means[i] = prior + lambda * (means[i] - prior);
    }
    return means;
  }

  public double[] steinBernoulli() {
    double[] means = new double[sum.length];
    for (int i = 0; i < means.length; ++i) {
      means[i] = sum[i] / tasks[i].length;
    }
    double norm = 0;
    double tr = 0;
    double lambdaMax = 0;
    for (int i = 0; i < sum.length; ++i) {
      double diff = means[i];
      lambdaMax = Math.max(lambdaMax, 1.0 / 4 / tasks[i].length);
      norm += diff * diff * tasks[i].length * 4;
      tr += 1.0 / 4 / tasks[i].length;
    }
    double lambda = norm > 0 ? 1 - (tr / lambdaMax - 2) / (norm) : 1.0;
    lambda = lambda > 0 ? lambda : 0;
    lambda = lambda < 1 ? lambda : 1;
    for (int i = 0; i < means.length; ++i) {
      means[i] = lambda * means[i];
    }
    return means;
  }


  public static double[] bernoulliMTA(double[] sum, double[] counts) {
    Vec means = new ArrayVec(sum.length);
    for (int i = 0; i < sum.length; ++i) {
      means.set(i, sum[i] / counts[i]);
    }
    Mx A = new VecBasedMx(sum.length, sum.length);
    for (int i = 0; i < sum.length; ++i) {
      for (int j = i + 1; j < sum.length; ++j) {
        final double dist = sum[i] / counts[i] - sum[j] / counts[j];
        A.set(i, j, dist > 1e-9 ? 2.0 / dist * dist : 2.0 * 1e18);
        A.set(j, i, dist > 1e-9 ? 2.0 / dist * dist : 2.0 * 1e18);
      }
    }

    double[] sigma = new double[sum.length];
    for (int i = 0; i < sigma.length; ++i) {
      final double p = means.get(i);
      sigma[i] = counts[i] > 1 ? p * (1 - p) / (counts[i] - 1) : p * (1 - p);
    }

    Mx L = MxTools.laplacian(A);
    Mx W = new VecBasedMx(L.rows(), L.columns());
    for (int row = 0; row < L.rows(); ++row) {
      for (int col = 0; col < L.columns(); ++col) {
        W.set(row, col, (row == col ? 1 : 0) + sigma[row] * L.get(row, col) / sum.length);
      }
    }
    Mx inverse = MxTools.inverse(W);
    return MxTools.multiply(inverse, means).toArray();
  }


  public static double[] bernoulliStein(double[] sum, double[] counts) {
    double[] means = new double[sum.length];
    for (int i = 0; i < means.length; ++i) {
      means[i] = sum[i] / counts[i];
    }
    double norm = 0;
    double tr = 0;
    double lambdaMax = 0;
    for (int i = 0; i < sum.length; ++i) {
      double diff = means[i];
      lambdaMax = Math.max(lambdaMax, 1.0 / 4 / counts[i]);
      norm += diff * diff * counts[i] * 4;
      tr += 1.0 / 4 / counts[i];
    }
    double lambda = norm > 0 ? 1 - (tr / lambdaMax - 2) / (norm) : 1.0;
    lambda = lambda > 0 ? lambda : 0;
    lambda = lambda < 1 ? lambda : 1;
    for (int i = 0; i < means.length; ++i) {
      means[i] = lambda * means[i];
    }
    return means;
  }


  public static Vec stein(Vec means) {
    double sigma = 1;
    Vec js = copy(means);
    scale(js, (1 - (js.dim() - 2) * sigma * sigma / sqr(norm(js))));
    return js;
  }


  public static double[] bernoulliConst(double[] weights) {
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < weights.length; ++i) {
      final double p = weights[i];
      min = p < min ? p : min;
      max = p > max ? p : max;
    }
    double a = Math.abs(min - max) > 1e-9 ? 2.0 / (weights.length * (min - max) * (min - max)) : 2e18;
    double[] Z = new double[weights.length];
    double z[] = new double[weights.length];
    for (int i = 0; i < weights.length; ++i) {
      Z[i] = 1 + a * weights.length;
      Z[i] = 1.0 / Z[i];
      z[i] = a * (weights[i] * (1 - weights[i])) / weights.length;
    }

    final double[] Zz = new double[weights.length];
    double[] result = new double[weights.length];
    double denum = 1;
    double num = 0;
    for (int i = 0; i < weights.length; ++i) {
      Zz[i] = Z[i] * z[i];
      denum -= Zz[i];
      result[i] = Z[i] * weights[i];
      num += result[i];
    }
    for (int i = 0; i < result.length; ++i) {
      result[i] += num * Zz[i] / denum;
    }

    return result;
  }


  public Mx bernoulliSimilarity() {
    double[] upperBounds = new double[sum.length];
    double[] lowerBounds = new double[sum.length];
    double[] means = new double[sum.length];
    for (int i = 0; i < sum.length; ++i) {
      final double n = tasks[i].length;
      final double p = sum[i] / n;
      means[i] = p;
      final double z = 0;
//      upperBounds[i] = p + 1.96 * Math.sqrt( p * (1-p) / n);
      upperBounds[i] = 1 / (1 + z * z / n);
      upperBounds[i] *= (p + z * z / (2 * n) + z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)));
      upperBounds[i] = Math.min(upperBounds[i], 1);
//      lowerBounds[i] =Math.max(p - 1.96 * Math.sqrt( p * (1-p) / n), 0);
      lowerBounds[i] = 1 / (1 + z * z / n);
      lowerBounds[i] *= (p + z * z / (2 * n) - z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)));
      lowerBounds[i] = Math.max(lowerBounds[i], 0);
    }
    Mx A = new VecBasedMx(sum.length, sum.length);
    for (int i = 0; i < sum.length; ++i) {
      for (int j = i + 1; j < sum.length; ++j) {
        final double upDist = Math.max(Math.abs(upperBounds[j] - lowerBounds[i]), Math.abs(upperBounds[j] - lowerBounds[i]));
//        final double upDist =0.5*(Math.abs(upperBounds[j] - lowerBounds[i])+Math.abs(upperBounds[j] - lowerBounds[i])));
        A.set(i, j, upDist > 0 ? 2.0 / upDist * upDist : 2.0 * 1e12);
        A.set(j, i, upDist > 0 ? 2.0 / upDist * upDist : 2.0 * 1e12);
      }
    }
    return A;
  }

  public Vec oracle(Mx A) {
    Mx L = MxTools.laplacian(A);
    Mx W = new VecBasedMx(L.rows(), L.columns());
    for (int row = 0; row < L.rows(); ++row) {
      for (int col = 0; col < L.columns(); ++col) {
        W.set(row, col, (row == col ? 1 : 0) + sigma[row] * L.get(row, col) / tasks.length);
      }
    }
    Vec means = new ArrayVec(sum.length);
    for (int i = 0; i < sum.length; ++i) {
      means.set(i, sum[i] / tasks[i].length);
    }
    Mx inverse = MxTools.inverse(W);
    return MxTools.multiply(inverse, means);
  }

  public double[] classic() {
    double[] result = new double[tasks.length];
    for (int i = 0; i < tasks.length; ++i) {
      result[i] = sum[i] / tasks[i].length;
    }
    return result;
  }

  public double[] mtaConst() {
    double a = 0;
    for (int i = 0; i < sum.length; ++i) {
      for (int j = i + 1; j < sum.length; ++j) {
        final double val = sum[i] / tasks[i].length - sum[j] / tasks[j].length;
        a += val * val;
      }
    }
    a = a > 0 ? tasks.length * (tasks.length - 1) / a : tasks.length * (tasks.length - 1) / 1e-12;
    double[] Z = new double[tasks.length];
    double z[] = new double[tasks.length];
    for (int i = 0; i < tasks.length; ++i) {
      Z[i] = 1 + a * sigma[i];
      Z[i] = 1.0 / Z[i];
      z[i] = a * sigma[i] / tasks.length;
    }

    final double[] Zz = new double[tasks.length];
    double[] result = new double[tasks.length];
    double denum = 1;
    double num = 0;
    for (int i = 0; i < tasks.length; ++i) {
      Zz[i] = Z[i] * z[i];
      denum -= Zz[i];
      result[i] = Z[i] * (sum[i] / tasks[i].length);
      num += result[i];
    }
    for (int i = 0; i < result.length; ++i) {
      result[i] += num * Zz[i] / denum;
    }

    return result;
  }

  public double[] mtaMiniMax() {
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < sum.length; ++i) {
      final double p = sum[i] / tasks[i].length;
      min = p < min ? p : min;
      max = p > max ? p : max;
    }
    double a = Math.abs(min - max) > 1e-9 ? 2.0 / (tasks.length * (min - max) * (min - max)) : 2e18;
    double[] Z = new double[tasks.length];
    double z[] = new double[tasks.length];
    for (int i = 0; i < tasks.length; ++i) {
      Z[i] = 1 + a * sigma[i];
      Z[i] = 1.0 / Z[i];
      z[i] = a * sigma[i] / tasks.length;
    }

    final double[] Zz = new double[tasks.length];
    double[] result = new double[tasks.length];
    double denum = 1;
    double num = 0;
    for (int i = 0; i < tasks.length; ++i) {
      Zz[i] = Z[i] * z[i];
      denum -= Zz[i];
      result[i] = Z[i] * (sum[i] / tasks[i].length);
      num += result[i];
    }
    for (int i = 0; i < result.length; ++i) {
      result[i] += num * Zz[i] / denum;
    }
    return result;
  }


  public double[] mtaMiniMaxBernoulli() {
    double min = 0;
    double max = 1;
    double a = Math.abs(min - max) > 1e-9 ? 2.0 / (tasks.length * (min - max) * (min - max)) : 2e18;
    double[] Z = new double[tasks.length];
    double z[] = new double[tasks.length];
    for (int i = 0; i < tasks.length; ++i) {
      Z[i] = 1 + a * sigma[i];
      Z[i] = 1.0 / Z[i];
      z[i] = a * sigma[i] / tasks.length;
    }

    final double[] Zz = new double[tasks.length];
    double[] result = new double[tasks.length];
    double denum = 1;
    double num = 0;
    for (int i = 0; i < tasks.length; ++i) {
      Zz[i] = Z[i] * z[i];
      denum -= Zz[i];
      result[i] = Z[i] * (sum[i] / tasks[i].length);
      num += result[i];
    }
    for (int i = 0; i < result.length; ++i) {
      result[i] += num * Zz[i] / denum;
    }
    return result;
  }
}


