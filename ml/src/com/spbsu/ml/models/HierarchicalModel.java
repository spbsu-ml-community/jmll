package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.TransJoin;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalModel extends MultiClassModel {
  TIntObjectHashMap<HierarchicalModel> children = new TIntObjectHashMap<HierarchicalModel>();
  TIntList classLabels = new TIntArrayList();

  public HierarchicalModel(Func[] dirs, TIntList classLabels) {
    super(dirs);
    if (classLabels.size() == dirs.length + 1)
      this.classLabels = classLabels;
    else
      throw new IllegalArgumentException("Classes count = " + (dirs.length + 1) + ", found " + classLabels.size() + " class labels");

  }

  public void addChildren(HierarchicalModel child, int catId) {
    children.put(catId, child);
  }

  public boolean isLeaf() {
    return children.size() == 0;
  }

  public int bestClass(Vec x) {
    int c = super.bestClass(x);
    int label = classLabels.get(c);
    return children.containsKey(label)? children.get(label).bestClass(x) : label;
  }

  public Vec bestClassAll(Mx data) {
    Vec result = new ArrayVec(data.rows());
    for (int i = 0; i < data.rows(); i++) {
      result.set(i, bestClass(data.row(i)));
    }
    return result;
  }

  public Vec probs(Vec x) {
    return null;
  }
}
