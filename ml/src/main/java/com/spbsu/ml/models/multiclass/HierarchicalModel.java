package com.spbsu.ml.models.multiclass;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
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
  protected final TIntObjectMap<HierarchicalModel> label2childModel;

  public HierarchicalModel(final MCModel basedOn, final TIntList classLabels) {
    this.basedOn = basedOn;
    this.classLabels = classLabels;
    this.label2childModel = new TIntObjectHashMap<>(classLabels.size());
  }

  public void addChild(final HierarchicalModel child, final int label) {
    label2childModel.put(label, child);
  }

  @Nullable
  public HierarchicalModel getChild(final int label) {
    return label2childModel.get(label);
  }

  @Override
  public int countClasses() {
    return basedOn.countClasses();
  }

  @Override
  public Vec probs(final Vec x) {
    return basedOn.probs(x);
  }

  @Override
  public int bestClass(final Vec x) {
    final int c = basedOn.bestClass(x);
    final int label = classLabels.get(c);
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
