package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TIntList;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalModel extends MultiClassModel {
  protected final TIntList classLabels;
  protected final TIntObjectHashMap<HierarchicalModel> label2childModel = new TIntObjectHashMap<HierarchicalModel>();

  public HierarchicalModel(Func[] dirs, TIntList classLabels) {
    super(dirs);
    if (classLabels.size() == dirs.length + 1)
      this.classLabels = classLabels;
    else
      throw new IllegalArgumentException("Classes count = " + (dirs.length + 1) + ", found " + classLabels.size() + " class labels");
  }

  public void addChild(HierarchicalModel child, int catId) {
    label2childModel.put(catId, child);
  }

  @Override
  public int bestClass(Vec x) {
    int c = super.bestClass(x);
    int label = classLabels.get(c);
    return label2childModel.containsKey(label)? label2childModel.get(label).bestClass(x) : label;
  }

  @Override
  public String toString() {
    return "splits " + classLabels.toString() + " classes, has child models for " +
        Arrays.toString(label2childModel.keys());
  }
}
