package com.expleague.fm;

import com.expleague.ml.GridTest;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.FMTrainingWorkaround;

public class FactorizationMachineTest extends GridTest {
  public void testFMRun() {
    final FMTrainingWorkaround fm = new FMTrainingWorkaround("r", "1,1,8", "10");
    fm.fit(learn.vecData(), learn.target(L2.class));
  }
}
