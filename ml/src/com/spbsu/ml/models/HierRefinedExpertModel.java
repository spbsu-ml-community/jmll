package com.spbsu.ml.models;

import com.spbsu.ml.Func;
import gnu.trove.list.TIntList;

/**
 * User: qdeee
 * Date: 18.04.14
 */
public class HierRefinedExpertModel extends HierarchicalModel {
  protected final HierJoinedBinClassModel bottomUpModel;

  public HierRefinedExpertModel(Func[] dirs, TIntList classLabels, HierJoinedBinClassModel bottomUpModel) {
    super(dirs, classLabels);
    this.bottomUpModel = bottomUpModel;
  }


}
