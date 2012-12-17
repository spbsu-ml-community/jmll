package com.spbsu.ml.methods;

import Jama.Matrix;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.loss.LossFunction;
import com.spbsu.ml.models.NormalizedLinearModel;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 27.12.10
 * Time: 18:04
 */
public class LARSMethod implements MLMethod {
  private static final double E = 1e-7;
  private class Direction {
    double sign;
    int index;
    private Direction(double sign, int index) {
      this.sign = sign;
      this.index = index;
    }
  }

  public NormalizedLinearModel fit(DataSet orig, LossFunction loss) {
    if (loss.getClass() != L2Loss.class)
      throw new IllegalArgumentException("LASSO can not be applied to loss other than l2");
    final int featuresCount = orig.xdim();
    final double[] betas = new double[featuresCount];
    double[] values = new double[orig.power()];
    final DataTools.NormalizationProperties props = new DataTools.NormalizationProperties();
    DataSet learn = DataTools.normalize(orig, DataTools.NormalizationType.SCALE, props);
    {
      final DSIterator it = learn.iterator();
      for (int i = 0; i < values.length; i++) {
        it.advance();
        values[i] = it.y();
      }
    }

    for (int t = 0; t < featuresCount; t++) {
      double[] correlations = new double[featuresCount];
      double bestCorr;
      final List<Direction> selectedDirections = new ArrayList<Direction>(featuresCount);
      {
        final DSIterator it = learn.iterator();
        int index = 0;
        while (it.advance()) {
          for (int i = 0; i < correlations.length; i++) {
            correlations[i] += it.x(i) * values[index];
          }
          index++;
        }
        bestCorr = Math.abs(correlations[0]);
        selectedDirections.add(new Direction(Math.signum(correlations[0]), 0));
        for (int i = 1; i < correlations.length; i++) {
          final double current = Math.abs(correlations[i]);
          final double diff = current - bestCorr;
          if (diff > E) {
            bestCorr = current;
            selectedDirections.clear();
            selectedDirections.add(new Direction(Math.signum(correlations[i]), i));
          }
          else if (diff < E && diff > -E) {
            selectedDirections.add(new Direction(Math.signum(correlations[i]), i));
          }
        }
      }
      final Matrix inverseCo;
      {
        final Matrix covariance = new Matrix(selectedDirections.size(), selectedDirections.size());
        final DSIterator it = learn.iterator();
        while (it.advance()) {
          for (int i = 0; i < selectedDirections.size(); i++) {
            final Direction d1 = selectedDirections.get(i);
            for (int j = 0; j < selectedDirections.size(); j++) {
              final Direction d2 = selectedDirections.get(j);
              covariance.set(i, j, covariance.get(i, j) + it.x(d1.index) * it.x(d2.index));
            }
          }
        }
        inverseCo = covariance.inverse();
      }
      final Matrix vec1 = new Matrix(selectedDirections.size(), 1, 1);
      double norm = Math.sqrt(vec1.transpose().times(inverseCo.times(vec1)).get(0, 0));
      Matrix w = inverseCo.times(vec1).times(norm);
      double[] equiangular = new double[learn.power()];
      {
        final DSIterator it = learn.iterator();
        int index = 0;
        while (it.advance()) {
          for (int i = 0; i < selectedDirections.size(); i++) {
            final Direction direction = selectedDirections.get(i);
            equiangular[index] += direction.sign * it.x(direction.index) * w.get(i, 0);
          }
          index++;
        }
      }

      double[] a = new double[featuresCount];
      {
        final DSIterator it = learn.iterator();
        int index = 0;
        while (it.advance()) {
          final double eqaComponent = equiangular[index];
          for (int i = 0; i < featuresCount; i++) {
            a[i] += it.x(i) * eqaComponent;
          }
          index++;
        }
      }

      double step = Double.MAX_VALUE;
      {
        for (final Direction direction : selectedDirections) {
          int j = direction.index;
          final double s1 = (bestCorr - correlations[j])/(norm - a[j]);
          final double s2 = (bestCorr + correlations[j])/(norm + a[j]);
          if (s1 > 0)
            step = Math.min(s1, step);
          if (s2 > 0)
            step = Math.min(s2, step);
        }
      }

      for (final Direction direction : selectedDirections) {
        final double signedStep = step * direction.sign;
        betas[direction.index] += signedStep;
      }
      {
        final DSIterator it = learn.iterator();
        for (int i = 0; i < values.length; i++) {
          it.advance();
          for (final Direction direction : selectedDirections) {
            final double signedStep = step * direction.sign;
            values[i] -= signedStep * it.x(direction.index);
          }
        }
      }
    }
    return new NormalizedLinearModel(new ArrayVec(betas), props);
  }
}
