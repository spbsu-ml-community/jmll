package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Func;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalModel extends MultiClassModel {
  TIntList classLabels = new TIntArrayList();
  TIntObjectHashMap<HierarchicalModel> label2childrenModel = new TIntObjectHashMap<HierarchicalModel>();

  public HierarchicalModel(Func[] dirs, TIntList classLabels) {
    super(dirs);
    if (classLabels.size() == dirs.length + 1)
      this.classLabels = classLabels;
    else
      throw new IllegalArgumentException("Classes count = " + (dirs.length + 1) + ", found " + classLabels.size() + " class labels");

  }

  public void addChildren(HierarchicalModel child, int catId) {
    label2childrenModel.put(catId, child);
  }

  public boolean isLeaf() {
    return label2childrenModel.size() == 0;
  }

  public int bestClass(Vec x) {
    int c = super.bestClass(x);
    int label = classLabels.get(c);
    return label2childrenModel.containsKey(label)? label2childrenModel.get(label).bestClass(x) : label;
  }

  public Vec bestClassAll(Mx data) {
    Vec result = new ArrayVec(data.rows());
    for (int i = 0; i < data.rows(); i++) {
      result.set(i, bestClass(data.row(i)));
    }
    return result;
  }
}
