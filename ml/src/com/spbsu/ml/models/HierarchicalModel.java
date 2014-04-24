package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Func;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalModel extends MultiClassModel {
  protected final TIntList classLabels;
  protected final TIntObjectHashMap<HierarchicalModel> label2childrenModel = new TIntObjectHashMap<HierarchicalModel>();

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

  public TIntObjectIterator<HierarchicalModel> getModelsIterator() {
    return label2childrenModel.iterator();
  }

  @Override
  public int bestClass(Vec x) {
    int c = super.bestClass(x);
    int label = classLabels.get(c);
    return label2childrenModel.containsKey(label)? label2childrenModel.get(label).bestClass(x) : label;
  }
}
