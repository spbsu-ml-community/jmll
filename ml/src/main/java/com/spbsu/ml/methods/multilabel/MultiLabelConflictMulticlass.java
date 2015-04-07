package com.spbsu.ml.methods.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxBuilder;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.impl.mx.MxByRowsBuilder;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multilabel.ClassicMultiLabelLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.models.multilabel.ConflictThresholdMultiLabelModel;

/**
 * User: qdeee
 * Date: 22.03.15
 */
public class MultiLabelConflictMulticlass implements VecOptimization<ClassicMultiLabelLoss> {
  private final VecOptimization<BlockwiseMLLLogit> weakMultiClass;
  private final double threshold;
  private final boolean allZeroesClassEnabled;

  public MultiLabelConflictMulticlass(final VecOptimization<BlockwiseMLLLogit> weakMultiClass, final double threshold, final boolean allZeroesClassEnabled) {
    this.weakMultiClass = weakMultiClass;
    this.threshold = threshold;
    this.allZeroesClassEnabled = allZeroesClassEnabled;
  }

  @Override
  public ConflictThresholdMultiLabelModel fit(final VecDataSet learn, final ClassicMultiLabelLoss classicMultiLabelLoss) {
    final Mx sourceFeatures = learn.data();
    final Mx sourceTargets = classicMultiLabelLoss.getTargets();
    final Pair<Vec, Mx> conflictData = createConflictData(sourceTargets, sourceFeatures, allZeroesClassEnabled);

    final Mx featuresWithDuplicate = conflictData.getSecond();
    final Vec conflictTarget = conflictData.getFirst();

    final VecDataSet ds = new VecDataSetImpl(featuresWithDuplicate, null);
    final BlockwiseMLLLogit mllLogit = new BlockwiseMLLLogit(conflictTarget, null);

    final MCModel mcModel = (MCModel) weakMultiClass.fit(ds, mllLogit);
    return new ConflictThresholdMultiLabelModel(mcModel, threshold, allZeroesClassEnabled);
  }

  private static Pair<Vec, Mx> createConflictData(final Mx targets, final Mx features, final boolean allZeroesClassEnabled) {
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
