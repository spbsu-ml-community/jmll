package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.ml.methods.MultiClass;
import com.expleague.commons.func.Factory;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MultiClassSplitBuilder implements Factory<MultiClass> {
  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String localName = "SatL2";
  private boolean printErr = false;

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  public void setLocal(final String localName) {
    this.localName = localName;
  }

  public void setOut(final boolean printErr) {
    this.printErr = printErr;
  }

  @Override
  public MultiClass create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    return new MultiClass(weak, (Class<? extends L2>) DataTools.targetByName(localName), printErr);
  }}
