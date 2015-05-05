package com.spbsu.ml.methods.multiclass.spoc;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxIterator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;

/**
 * User: qdeee
 * Date: 14.11.14
 */
public class CMLHelper {
  public static final double MX_IGNORE_THRESHOLD = 0.1;

  public static boolean checkConstraints(final Mx B) {
    for (int l = 0; l < B.columns(); l++) {
      double sumPositive = 0;
      double sumNegative = 0;
      for (int k = 0; k < B.rows(); k++) {
        final double code = B.get(k, l);
        final double absCode = Math.abs(code);
        if (absCode > 1)
          return false;
        sumPositive += absCode + code;
        sumNegative += absCode - code;
      }
      if (sumPositive < 2 || sumNegative < 2)
        return false;
    }
    for (int k = 0; k < B.rows(); k++) {
      final double sum = VecTools.l1(B.row(k));
      if (sum < 1)
        return false;
    }
    return true;
  }

  static void normalizeMx(final Mx codingMatrix, final double mxIgnoreThreshold) {
    for (final MxIterator iter = codingMatrix.nonZeroes(); iter.advance(); ) {
      final double value = iter.value();
      if (Math.abs(value) > mxIgnoreThreshold)
        iter.setValue(Math.signum(value));
      else
        iter.setValue(0.0);
    }
  }

  //check pairwise columns independence
  public static boolean checkColumnsIndependence(final Mx B) {
    for (int j1 = 0; j1 < B.columns(); j1++) {
      final Vec col1 = B.col(j1);
      final double norm1 = VecTools.norm(col1);
      if (norm1 == 0)
        return false;
      for (int j2 = j1 + 1; j2 < B.columns(); j2++) {
        final Vec col2 = B.col(j2);
        final double norm2 = VecTools.norm(col2);
        if (norm2 == 0)
          return false;
        final double cosine = VecTools.multiply(col1, col2) / (norm1 * norm2);
        if (Math.abs(cosine) > 0.999)
          return false;
      }
    }
    return true;
  }
}
