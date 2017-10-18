package com.expleague.ml.methods.multilabel;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.impl.mx.MxByRowsBuilder;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.models.multiclass.MCModel;
import com.expleague.commons.math.vectors.MxBuilder;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.util.Pair;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multilabel.ClassicMultiLabelLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.multilabel.ConflictThresholdMultiLabelModel;

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
