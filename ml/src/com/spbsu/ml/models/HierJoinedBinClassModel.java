package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.Func;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TIntList;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.Iterator;

/**
 * User: qdeee
 * Date: 14.04.14
 */
public class HierJoinedBinClassModel extends JoinedBinClassModel {
  /**
   * Labels of classes that this model could return.
   */
  protected final TIntList classLabels;

  protected final TIntObjectHashMap<HierJoinedBinClassModel> label2childModel = new TIntObjectHashMap<HierJoinedBinClassModel>();

  public HierJoinedBinClassModel(final Func[] dirs, final TIntList classLabels) {
    super(dirs);
    if (classLabels.size() == dirs.length) {
      this.classLabels = classLabels;
    }
    else {
      throw new IllegalArgumentException("Classes count (" + dirs.length + ") does not equal to labels count (" + classLabels.size() + ")");
    }
  }

  public void addChildren(HierJoinedBinClassModel child, int classId) {
    label2childModel.put(classId, child);
  }

  public HierJoinedBinClassModel getModelByLabel(int label) {
    return label2childModel.get(label);
  }

  @Override
  public int bestClass(Vec x) {
    int c = super.bestClass(x);
    return classLabels.get(c);
  }

  @Override
  public Vec trans(final Vec x) {
    final Vec trans = super.trans(x);
    for (int i = 0; i < classLabels.size(); i++) {
      int label = classLabels.get(i);
      HierJoinedBinClassModel model = label2childModel.get(label);
      if (model != null) {
        final double val = VecTools.sum(model.trans(x));
        trans.adjust(i, val);
      }
    }
    return trans;
  }
}
