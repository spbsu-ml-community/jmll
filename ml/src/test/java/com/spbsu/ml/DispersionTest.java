package com.spbsu.ml;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
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
public abstract class DispersionTest extends TestCase {
  public void testStein() {
    final Random rng = new FastRandom();
    for (int n = 1; n < 100; n++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      for (int k = 0; k < 1000; k++) {
        final Vec sum = new ArrayVec(3);
        final Vec m = new ArrayVec(new double[]{2 * rng.nextDouble(), 2 * rng.nextDouble(), 2 * rng.nextDouble()});
        final double sigma = 1;

        for (int i = 0; i < n; i++) {
          for (int t = 0; t < sum.dim(); t++) {
            sum.adjust(t, sigma * rng.nextGaussian() + m.get(t));
          }
        }
        final Vec naive = copy(sum); scale(naive, 1./n);
        final Vec js = copy(naive); scale(js, (1-(js.dim() - 2)*sigma*sigma/sqr(norm(sum))));

        sumAvgNaive += distance(naive, m)/Math.sqrt(m.dim());
        sumAvgJS += distance(js, m)/Math.sqrt(m.dim());
      }
      System.out.println(n + "\t" + sumAvgNaive/1000 + "\t" + sumAvgJS/1000);
    }
  }

  public void testSteinBernoulli() {
    final Random rng = new FastRandom();
    for (int n = 1; n < 100; n++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      for (int k = 0; k < 1000; k++) {
        final Vec sum = new ArrayVec(3);
        final Vec m = new ArrayVec(rng.nextDouble(), rng.nextDouble(), rng.nextDouble());
        final double sigma = 1;

        for (int i = 0; i < n; i++) {
          for (int t = 0; t < sum.dim(); t++) {
            sum.adjust(t, rng.nextDouble() > m.get(t) ? 0 : 1);
          }
        }
        final Vec naive = copy(sum); scale(naive, 1./n);
        final Vec js = copy(sum); scale(js, (1-(js.dim() - 2)*sigma*sigma/sqr(norm(sum)))/n);

        sumAvgNaive += distance(naive, m)/Math.sqrt(m.dim());
        sumAvgJS += distance(js, m)/Math.sqrt(m.dim());
      }
      System.out.println(n + "\t" + sumAvgNaive/1000 + "\t" + sumAvgJS/1000);
    }
  }

  public void test1FoldClear() {
    final Random rng = new FastRandom();
    double sumAvgNaive = 0;
    double sumAvgJS = 0;
    double sumDGP = 0;
    for (int k = 0; k < 1000; k++) {
      final Vec set = new ArrayVec(3);
      final double m_0 = rng.nextDouble();
      final Vec m = new ArrayVec(m_0, m_0, m_0);
      final double sigma = 1;

      for (int t = 0; t < set.dim(); t++) {
        set.set(t, sigma * rng.nextGaussian() + m.get(t));
      }
      final double naive;
      {
        final Vec estimate = new ArrayVec(set.dim());
        naive = MathTools.meanNaive(set);
        fill(estimate, naive);
        sumAvgNaive += distance(estimate, m)/Math.sqrt(m.dim());
      }
      final double js;
      {
        final Vec estimate = new ArrayVec(set.dim());
        js = MathTools.meanJS1(set, sigma);
        fill(estimate, js);
        sumAvgJS += distance(estimate, m)/Math.sqrt(m.dim());
      }
      {
        final Vec estimate = new ArrayVec(set.dim());
        final double dgp = MathTools.meanDropFluctuations(set);
        fill(estimate, dgp);
        sumDGP += distance(estimate, m)/Math.sqrt(m.dim());
      }
    }
    System.out.println("\t" + sumAvgNaive/1000 + "\t" + sumAvgJS/1000 + "\t" + sumDGP/1000);
  }

  public void test1FoldErr10() {
    final Random rng = new FastRandom();
    for (int m = 3; m < 100; m++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumDGP = 0;
      for (int k = 0; k < 1000; k++) {
        final Vec set = new ArrayVec(m);
        final double m_0 = rng.nextDouble() * 2;
        final Vec e = new ArrayVec(set.dim());
        for (int i = 0; i < e.dim(); i++) {
          e.set(i, rng.nextDouble() > 0.3 ? m_0 : rng.nextDouble() * 2);
        }
        final double sigma = 1;

        for (int t = 0; t < set.dim(); t++) {
          set.set(t, sigma * rng.nextGaussian() + e.get(t));
        }
        final double naive;
        {
          final Vec estimate = new ArrayVec(set.dim());
          naive = MathTools.meanNaive(set);
          fill(estimate, naive);
          sumAvgNaive += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0))/Math.sqrt(e.dim());
        }
        final double js;
        {
          final Vec estimate = new ArrayVec(set.dim());
          js = MathTools.meanJS1(set, sigma);
          fill(estimate, js);
          sumAvgJS += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0))/Math.sqrt(e.dim());
        }
        {
          final Vec estimate = new ArrayVec(set.dim());
          final double dgp = MathTools.meanDropFluctuations(set);
          fill(estimate, dgp);
          sumDGP += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0))/Math.sqrt(e.dim());
        }
      }
      System.out.println(m + "\t" + sumAvgNaive/1000 + "\t" + sumAvgJS/1000 + "\t" + sumDGP/1000);
    }
  }

  public void test1FoldErrBig10() {
    final Random rng = new FastRandom();
    for (int m = 3; m < 100; m++) {
      double sumAvgNaive = 0;
      double sumAvgJS = 0;
      double sumDGP = 0;
      for (int k = 0; k < 1000; k++) {
        final Vec set = new ArrayVec(m);
        final double m_0 = rng.nextDouble() * 2 - 1;
        final Vec e = new ArrayVec(set.dim());
        for (int t = 0; t < e.dim(); t++) {
          e.set(t, rng.nextDouble() > 0.01 ? m_0 : (rng.nextBoolean() ? 1 : -1)/(1 - rng.nextDouble()) * 2);
        }
        final double sigma = 1;

        for (int t = 0; t < set.dim(); t++) {
          set.set(t, sigma * rng.nextGaussian() + e.get(t));
        }
        final double naive;
        {
          final Vec estimate = new ArrayVec(set.dim());
          naive = MathTools.meanNaive(set);
          fill(estimate, naive);
          sumAvgNaive += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0))/Math.sqrt(e.dim());
        }
        final double js;
        {
          final Vec estimate = new ArrayVec(set.dim());
          js = MathTools.meanJS1(set, sigma);
          fill(estimate, js);
          sumAvgJS += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0))/Math.sqrt(e.dim());
        }
        {
          final Vec estimate = new ArrayVec(set.dim());
          final double dgp = MathTools.meanDropFluctuations(set);
          fill(estimate, dgp);
          sumDGP += distance(estimate, fill(new ArrayVec(estimate.dim()), m_0))/Math.sqrt(e.dim());
        }
      }
      System.out.println(m + "\t" + sumAvgNaive/1000 + "\t" + sumAvgJS/1000 + "\t" + sumDGP/1000);
    }
  }
}
