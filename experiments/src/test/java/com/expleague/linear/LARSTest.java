package com.expleague.linear;

import com.expleague.ml.GridTest;
import com.expleague.ml.func.NormalizedLinear;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.LARSMethod;

public class LARSTest extends GridTest {
  public void testLARS() {
    final LARSMethod lars = new LARSMethod();
//    lars.addListener(modelPrinter);
    final Class<L2> targetClass = L2.class;
    final L2 target = learn.target(targetClass);
    final NormalizedLinear model = lars.fit(learn.vecData(), target);
    System.out.println(validate.target(L2.class).value(model.transAll((validate.vecData()).data())));
  }
}
