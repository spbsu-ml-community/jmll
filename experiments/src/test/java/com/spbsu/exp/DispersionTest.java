package com.spbsu.exp;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.methods.MTA;
import junit.framework.TestCase;

import java.util.Random;

import static com.spbsu.commons.math.MathTools.sqr;
import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Stein paradox experiments.
 * User: solar
 * Date: 16.04.14
 * Time: 13:24
 */
public class DispersionTest extends TestCase {
  public void testStein() {
    Random rng = new FastRandom();
    for (int n = 3; n < 1000; n++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumAvgJS2 = 0;
      double sumAvgMta = 0;
      double sumAvgMtaMiniMax = 0;
      double sumAvgMtaOrcale = 0;
      for (int k = 0; k < 1000; k++) {
        final int mdim = 15;
        Vec sum = new ArrayVec(mdim);
        Vec m = new ArrayVec(mdim);
//        Vec m = new ArrayVec(new double[]{2 * rng.nextDouble(), 2 * rng.nextDouble(), 2 * rng.nextDouble()});
        for (int i = 0; i < mdim; ++i) {
          m.set(i, rng.nextDouble() * 2);
        }
        Mx A = new VecBasedMx(mdim, mdim);
        for (int i = 0; i < mdim; ++i) {
          for (int j = i + 1; j < mdim; ++j) {
            double val = m.get(i) - m.get(j);
            val = 2 / (val * val);
            A.set(i, j, val);
            A.set(j, i, val);
          }
        }
        double sigma = 1;
        double[][] samples = new double[mdim][n];
        for (int i = 0; i < n; i++) {
          for (int t = 0; t < sum.dim(); t++) {
            samples[t][i] = sigma * rng.nextGaussian() + m.get(t);
            sum.adjust(t, samples[t][i]);
          }
        }
        Vec naive = copy(sum);
        scale(naive, 1. / n);
        Vec js = copy(naive);
        scale(js, (1 - (js.dim() - 2) * sigma * sigma / sqr(norm(sum))));
        MTA estimator = new MTA(samples);
        Vec mtaOrcale = estimator.oracle(A);
        Vec mta = new ArrayVec(estimator.mtaConst());
        sumAvgNaive += distance(naive, m) / Math.sqrt(m.dim());
        sumAvgJS += distance(js, m) / Math.sqrt(m.dim());
        sumAvgJS2 += distance(new ArrayVec(estimator.stein()), m) / Math.sqrt(m.dim());
        sumAvgMta += distance(mta, m) / Math.sqrt(m.dim());
        sumAvgMtaMiniMax += distance(new ArrayVec(new MTA(samples).mtaMiniMax()), m) / Math.sqrt(m.dim());
        sumAvgMtaOrcale += distance(mtaOrcale, m) / Math.sqrt(m.dim());
      }
      System.out.println(n + "\t" + sumAvgNaive / 1000 + "\t" + sumAvgJS / 1000 + "\t" + sumAvgJS2 / 1000 + "\t" + sumAvgMta / 1000 + "\t" + sumAvgMtaMiniMax / 1000 + "\t" + sumAvgMtaOrcale / 1000);
    }
  }

  public void testMtaBernoulli() {
    Random rng = new FastRandom();
    for (int n = 5; n < 1000; n++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumAvgJS2 = 0;
      double sumAvgMta = 0;
      double sumAvgMtaMiniMax = 0;
      double sumAvgMtaOracle = 0;
      int N = 1000;
      for (int k = 0; k < N; k++) {
        final int mdim = 10;
        Vec sum = new ArrayVec(mdim);
        Vec m = new ArrayVec(mdim);
        for (int i = 0; i < mdim; ++i) {
          m.set(i, rng.nextDouble());
        }

        Mx A = new VecBasedMx(mdim, mdim);
        for (int i = 0; i < mdim; ++i) {
          for (int j = i + 1; j < mdim; ++j) {
            double val = m.get(i) - m.get(j);
            val = 2 / (val * val);
            A.set(i, j, val);
            A.set(j, i, val);
          }
        }
        double sigma = 1.0;
        double[][] samples = new double[mdim][n];
        for (int i = 0; i < n; i++) {
          for (int t = 0; t < sum.dim(); t++) {
            samples[t][i] = rng.nextDouble() > m.get(t) ? 0 : 1;
            sum.adjust(t, samples[t][i]);
          }
        }
        Vec naive = copy(sum);
        scale(naive, 1. / n);
        Vec js = copy(naive);
        scale(js, (1 - (js.dim() - 2) * sigma / sqr(norm(sum))));
        MTA estimator = new MTA(samples);
        Vec mta = estimator.oracle(estimator.bernoulliSimilarity());
        Vec mtaOracle = estimator.oracle(A);
        sumAvgNaive += distance(naive, m) / Math.sqrt(m.dim());
        sumAvgJS += distance(js, m) / Math.sqrt(m.dim());
        sumAvgJS2 += distance(new ArrayVec(estimator.steinBernoulli()), m) / Math.sqrt(m.dim());
//        sumAvgJS2 += distance(MTA.bernoulliStein(sum, n), m) / Math.sqrt(m.dim());
        sumAvgMta += distance(mta, m) / Math.sqrt(m.dim());
        sumAvgMtaMiniMax += distance(new ArrayVec(new MTA(samples).mtaMiniMaxBernoulli()), m) / Math.sqrt(m.dim());
        sumAvgMtaOracle += distance(mtaOracle, m) / Math.sqrt(m.dim());
      }
      System.out.println(n + "\t" + sumAvgNaive / N + "\t" + sumAvgJS / N + "\t" + sumAvgJS2 / N + "\t" + sumAvgMta / N + "\t" + sumAvgMtaMiniMax / N + "\t" + sumAvgMtaOracle / N);
    }
  }


  public void testSteinBernoulli() {
    Random rng = new FastRandom();
    for (int n = 1; n < 100; n++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      for (int k = 0; k < 1000; k++) {
        Vec sum = new ArrayVec(3);
        Vec m = new ArrayVec(rng.nextDouble(), rng.nextDouble(), rng.nextDouble());
        double sigma = 1;

        for (int i = 0; i < n; i++) {
          for (int t = 0; t < sum.dim(); t++) {
            sum.adjust(t, rng.nextDouble() > m.get(t) ? 0 : 1);
          }
        }
        Vec naive = copy(sum);
        scale(naive, 1. / n);
        Vec js = copy(sum);
        scale(js, (1 - (js.dim() - 2) * sigma * sigma / sqr(norm(sum))) / n);

        sumAvgNaive += distance(naive, m) / Math.sqrt(m.dim());
        sumAvgJS += distance(js, m) / Math.sqrt(m.dim());
      }
      System.out.println(n + "\t" + sumAvgNaive / 1000 + "\t" + sumAvgJS / 1000);
    }
  }

  public void test1FoldClear() {
    Random rng = new FastRandom();
    double sumAvgNaive = 0;
    double sumAvgJS = 0;
    double sumDGP = 0;
    for (int k = 0; k < 1000; k++) {
      Vec set = new ArrayVec(3);
      final double m_0 = rng.nextDouble();
      Vec m = new ArrayVec(m_0, m_0, m_0);
      double sigma = 1;

      for (int t = 0; t < set.dim(); t++) {
        set.set(t, sigma * rng.nextGaussian() + m.get(t));
      }
      double naive;
      {
        Vec estimate = new ArrayVec(set.dim());
        naive = MathTools.meanNaive(set);
        fill(estimate, naive);
        sumAvgNaive += distance(estimate, m) / Math.sqrt(m.dim());
      }
      double js;
      {
        Vec estimate = new ArrayVec(set.dim());
        js = MathTools.meanJS1(set, sigma);
        fill(estimate, js);
        sumAvgJS += distance(estimate, m) / Math.sqrt(m.dim());
      }
      {
        Vec estimate = new ArrayVec(set.dim());
        double dgp = MathTools.meanDropFluctuations(set);
        fill(estimate, dgp);
        sumDGP += distance(estimate, m) / Math.sqrt(m.dim());
      }
    }
    System.out.println("\t" + sumAvgNaive / 1000 + "\t" + sumAvgJS / 1000 + "\t" + sumDGP / 1000);
  }

  public void test1FoldErr10() {
    Random rng = new FastRandom();
    for (int m = 3; m < 100; m++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumDGP = 0;
      for (int k = 0; k < 1000; k++) {
        Vec set = new ArrayVec(m);
        final double m_0 = rng.nextDouble() * 2;
        Vec e = new ArrayVec(set.dim());
        for (int i = 0; i < e.dim(); i++) {
          e.set(i, rng.nextDouble() > 0.3 ? m_0 : rng.nextDouble() * 2);
        }
        double sigma = 1;

        for (int t = 0; t < set.dim(); t++) {
          set.set(t, sigma * rng.nextGaussian() + e.get(t));
        }
        double naive;
        {
          Vec estimate = new ArrayVec(set.dim());
          naive = MathTools.meanNaive(set);
          fill(estimate, naive);
          sumAvgNaive += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0)) / Math.sqrt(e.dim());
        }
        double js;
        {
          Vec estimate = new ArrayVec(set.dim());
          js = MathTools.meanJS1(set, sigma);
          fill(estimate, js);
          sumAvgJS += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0)) / Math.sqrt(e.dim());
        }
        {
          Vec estimate = new ArrayVec(set.dim());
          double dgp = MathTools.meanDropFluctuations(set);
          fill(estimate, dgp);
          sumDGP += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0)) / Math.sqrt(e.dim());
        }
      }
      System.out.println(m + "\t" + sumAvgNaive / 1000 + "\t" + sumAvgJS / 1000 + "\t" + sumDGP / 1000);
    }
  }

  public void test1FoldErrBig10() {
    Random rng = new FastRandom();
    for (int m = 3; m < 100; m++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumDGP = 0;
      for (int k = 0; k < 1000; k++) {
        Vec set = new ArrayVec(m);
        final double m_0 = rng.nextDouble() * 2 - 1;
        Vec e = new ArrayVec(set.dim());
        for (int t = 0; t < e.dim(); t++) {
          e.set(t, rng.nextDouble() > 0.01 ? m_0 : (rng.nextBoolean() ? 1 : -1) / (1 - rng.nextDouble()) * 2);
        }
        double sigma = 1;

        for (int t = 0; t < set.dim(); t++) {
          set.set(t, sigma * rng.nextGaussian() + e.get(t));
        }
        double naive;
        {
          Vec estimate = new ArrayVec(set.dim());
          naive = MathTools.meanNaive(set);
          fill(estimate, naive);
          sumAvgNaive += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0)) / Math.sqrt(e.dim());
        }
        double js;
        {
          Vec estimate = new ArrayVec(set.dim());
          js = MathTools.meanJS1(set, sigma);
          fill(estimate, js);
          sumAvgJS += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0)) / Math.sqrt(e.dim());
        }
        {
          Vec estimate = new ArrayVec(set.dim());
          double dgp = MathTools.meanDropFluctuations(set);
          fill(estimate, dgp);
          sumDGP += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0)) / Math.sqrt(e.dim());
        }
      }
      System.out.println(m + "\t" + sumAvgNaive / 1000 + "\t" + sumAvgJS / 1000 + "\t" + sumDGP / 1000);
    }
  }
}
