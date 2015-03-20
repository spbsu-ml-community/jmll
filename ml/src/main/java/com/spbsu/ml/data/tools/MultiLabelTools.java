package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxBuilder;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.mx.MxByRowsBuilder;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.util.Pair;

/**
 * User: qdeee
 * Date: 20.03.15
 */
public final class MultiLabelTools {
  public static Pair<Vec, Mx> createConflictData(final Mx targets, final Mx features, final boolean allZeroesClassEnabled) {
    final MxBuilder mxBuilder = new MxByRowsBuilder();
    final VecBuilder vecBuilder = new VecBuilder();
    for (int i = 0; i < features.rows(); i++) {
      final Vec instanceTargets = targets.row(i);
      final Vec instanceFeatures = features.row(i);

      final VecIterator targetIter = instanceTargets.nonZeroes();
      boolean allZeroesTarget = true;
      while (targetIter.advance()) {
        final int targetIndex = targetIter.index();
        final double targetValue = targetIter.value();

        if (targetValue > 0) {
          allZeroesTarget = false;
          vecBuilder.append(targetIndex);
          mxBuilder.add(instanceFeatures);
        }
      }
      if (allZeroesTarget && allZeroesClassEnabled) {
        vecBuilder.append(instanceTargets.dim());
        mxBuilder.add(instanceFeatures);
      }
    }

    return Pair.create(vecBuilder.build(), mxBuilder.build());
  }
}
