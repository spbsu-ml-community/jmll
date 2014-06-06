package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Func;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 14.04.14
 */
public class HierJoinedBinClassAddMetaFeaturesModel extends JoinedBinClassModel {
  /**
   * Labels of classes that this model could return.
   */
  public final TIntList classLabels;

  protected final TIntObjectHashMap<HierJoinedBinClassAddMetaFeaturesModel> label2childModel = new TIntObjectHashMap<HierJoinedBinClassAddMetaFeaturesModel>();

  public HierJoinedBinClassAddMetaFeaturesModel(final Func[] dirs, final TIntList classLabels) {
    super(dirs);
    if (classLabels.size() == dirs.length) {
      this.classLabels = classLabels;
    }
    else {
      throw new IllegalArgumentException("Classes count (" + dirs.length + ") does not equal to labels count (" + classLabels.size() + ")");
    }
  }

  public void addChildren(HierJoinedBinClassAddMetaFeaturesModel child, int classId) {
    label2childModel.put(classId, child);
  }

  public HierJoinedBinClassAddMetaFeaturesModel getModelByLabel(int label) {
    return label2childModel.get(label);
  }

  @Override
  public double prob(final int classLabel, final Vec x) {
    final int pos = classLabels.indexOf(classLabel);
    if (pos == -1)
      throw new IllegalArgumentException("Invalid class label");
    return super.prob(pos, x);
  }

  @Override
  public int bestClass(Vec x) {
    final int c = super.bestClass(x);
    return classLabels.get(c);
  }

  //
  @Override
  public Vec trans(final Vec x) {
    final Vec result = new ArrayVec(dirs.length);
    final TDoubleList metafeatures = new TDoubleLinkedList();
    for (int i = 0; i < classLabels.size(); i++) {
      final int label = classLabels.get(i);
      final HierJoinedBinClassAddMetaFeaturesModel model = label2childModel.get(label);
      if (model != null) {
        final Vec childTrans = model.trans(x);
        final double sum = VecTools.sum(childTrans);
        result.adjust(i, sum);
        final Vec probs = model.probs(x);
        metafeatures.add(probs.toArray());
      }
    }
    final Vec extendX = VecTools.extendVec(x, metafeatures.toArray());
    final Vec selfTrans = super.trans(extendX);
    return VecTools.append(result, selfTrans);
  }

  @Override
  public String toString() {
    return "splits " + classLabels.toString() + " classes, has child models for " +
        Arrays.toString(label2childModel.keys());
  }
}
