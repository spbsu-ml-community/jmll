package com.spbsu.ml.methods;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.NormalizedLinear;
import com.spbsu.ml.loss.L2;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 27.12.10
 * Time: 18:04
 */
public class LARSMethod extends VecOptimization.Stub<L2> {
  private class Direction {
    double sign;
    int index;
    private Direction(final double sign, final int index) {
      this.sign = sign;
      this.index = index;
    }
  }

  @Override
  public NormalizedLinear fit(final VecDataSet origDS, final L2 loss) {
    final Mx orig = origDS.data();
    final int featuresCount = orig.columns();
    final Vec betas = new ArrayVec(featuresCount);
    final double avg = VecTools.sum(loss.target) / loss.xdim();
    final MxTools.NormalizationProperties props = new MxTools.NormalizationProperties();
    final Mx learn = MxTools.normalize(orig, MxTools.NormalizationType.SCALE, props);
    final Vec values = new ArrayVec(orig.rows());
    fill(values, -avg);
    append(values, loss.target);

    for (int t = 0; t < featuresCount; t++) {
      final Vec correlations = MxTools.multiply(MxTools.transpose(learn), values);
      final double bestCorr;
      final List<Direction> selectedDirections = new ArrayList<Direction>(featuresCount);
      {
        final int[] order = ArrayTools.sequence(0, correlations.dim());
        final Vec absCorr = abs(correlations);
        ArrayTools.parallelSort(absCorr.toArray(), order);
        bestCorr = Math.abs(correlations.get(order[0]));
        for (int i = 0; i < correlations.dim(); i++) {
          if (bestCorr - absCorr.get(order[i]) > MathTools.EPSILON)
            break;
          selectedDirections.add(new Direction(Math.signum(correlations.get(order[i])), order[i]));
        }
      }
      final Mx inverseCo;
      {
        final Mx covariance = new VecBasedMx(selectedDirections.size(), selectedDirections.size());
        for (int r = 0; r < learn.rows(); r++) {
          for (int i = 0; i < selectedDirections.size(); i++) {
            final Direction d1 = selectedDirections.get(i);
            for (int j = 0; j < selectedDirections.size(); j++) {
              final Direction d2 = selectedDirections.get(j);
              covariance.set(i, j, covariance.get(i, j) + learn.get(r, d1.index) * learn.get(r, d2.index));
            }
          }
        }
        inverseCo = MxTools.inverseCholesky(covariance);
      }
      final Vec ones = fill(new ArrayVec(selectedDirections.size()), 1.);
      final double norm = Math.sqrt(multiply(ones, MxTools.multiply(inverseCo, ones)));
      scale(ones, norm);
      final Vec w = MxTools.multiply(inverseCo, ones);
      final double[] equiangular = new double[learn.rows()];
      {
        for (int r = 0; r < learn.rows(); r++) {
          for (int i = 0; i < selectedDirections.size(); i++) {
            final Direction direction = selectedDirections.get(i);
            equiangular[r] += direction.sign * learn.get(r, direction.index) * w.get(i);
          }
        }
      }

      final double[] a = new double[featuresCount];
      {
        for (int r = 0; r < learn.rows(); r++) {
          final double eqaComponent = equiangular[r];
          for (int i = 0; i < featuresCount; i++) {
            a[i] += learn.get(r, i) * eqaComponent;
          }
        }
      }

      double step = Double.MAX_VALUE;
      {
        for (final Direction direction : selectedDirections) {
          final int j = direction.index;
          final double s1 = (bestCorr - correlations.get(j))/(norm - a[j]);
          final double s2 = (bestCorr + correlations.get(j))/(norm + a[j]);
          if (s1 > 0)
            step = Math.min(s1, step);
          if (s2 > 0)
            step = Math.min(s2, step);
        }
      }

      for (final Direction direction : selectedDirections) {
        final double signedStep = step * direction.sign;
        betas.adjust(direction.index, signedStep);
      }
      {
        for (int r = 0; r < learn.rows(); r++) {
          for (final Direction direction : selectedDirections) {
            final double signedStep = step * direction.sign;
            values.adjust(r, -signedStep * learn.get(r, direction.index));
          }
        }
      }
    }
    return new NormalizedLinear(avg, betas, props);
  }
}
