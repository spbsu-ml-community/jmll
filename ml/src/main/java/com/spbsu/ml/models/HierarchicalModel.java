package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.Trans;
import gnu.trove.list.TIntList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

/**
 * User: qdeee
 * Date: 06.02.14
 */
public class HierarchicalModel extends MCModel.Stub {
  protected final MCModel basedOn;
  protected final TIntList classLabels;
  protected final TIntObjectMap<HierarchicalModel> label2childModel = new TIntObjectHashMap<>();

  public HierarchicalModel(MCModel basedOn, TIntList classLabels) {
    this.basedOn = basedOn;
    this.classLabels = classLabels;
  }

  public void addChild(HierarchicalModel child, int label) {
    label2childModel.put(label, child);
  }

  @Override
  public Vec probs(final Vec x) {
    return basedOn.probs(x);
  }

  @Override
  public int bestClass(Vec x) {
    int c = basedOn.bestClass(x);
    int label = classLabels.get(c);
    return label2childModel.containsKey(label)? label2childModel.get(label).bestClass(x) : label;
  }

  @Override
  public Vec bestClassAll(final Mx x) {
    return transAll(x);
  }

  @Override
  public int dim() {
    return basedOn.dim();
  }

  @Override
  public String toString() {
    return "splits " + classLabels.toString() + " classes, has child models for " +
        Arrays.toString(label2childModel.keys());
  }
}
