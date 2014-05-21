package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.Func;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;

/**
 * User: qdeee
 * Date: 18.04.14
 */
public class HierRefinedExpertModel extends HierarchicalModel {
  public final HierJoinedBinClassAddMetaFeaturesModel bottomUpModel;

  public HierRefinedExpertModel(Func[] dirs, TIntList classLabels, HierJoinedBinClassAddMetaFeaturesModel bottomUpModel) {
    super(dirs, classLabels);
    this.bottomUpModel = bottomUpModel;
  }

  @Override
  public Vec trans(final Vec x) {
    final Vec bottomTopTrans = bottomUpModel.trans(x);
    final TDoubleList metafeatures = new TDoubleLinkedList();
    for (int i = 0; i < classLabels.size(); i++) {
      final HierJoinedBinClassAddMetaFeaturesModel model = bottomUpModel.label2childModel.get(classLabels.get(i));
      if (model != null) {
        final Vec probs = model.probs(x);
        metafeatures.add(probs.toArray());
      }
    }

    final Vec extendX = VecTools.extendVec(x, metafeatures.toArray());
    final Vec selfTrans = super.trans(extendX);
    return VecTools.sum(selfTrans, bottomTopTrans);
  }
}
