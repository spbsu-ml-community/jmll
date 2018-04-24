package com.expleague.ml.models.nn;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.generic.LogSoftMax;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SoftMaxTest {
  private static final int numClasses = 10;
  private static final FastRandom rng = new FastRandom();
  private static final double EPS = 1e-6;
  private static final int ROUNDS = 10_000;

  @Test
  public void simpleTest() {
    Vec arg = new ArrayVec(numClasses);
    Vec grad = new ArrayVec(numClasses);

    for (int testCls = 0; testCls < numClasses; testCls++) {
      final FuncC1 loss = new LogSoftMax(numClasses, testCls);

      for (int round = 0; round < ROUNDS; round++) {
        VecTools.fillUniform(arg, rng);
        loss.gradientTo(arg, grad);
        final double result = loss.value(arg);
        for (int cls = 0; cls < numClasses; cls++) {
          arg.adjust(cls, EPS);
          final double incResult = loss.value(arg);
          final double gradIdx = (incResult - result) / EPS;

//          System.out.println(gradIdx + " " + grad.get(cls));
          assertEquals("testing idx " + cls, gradIdx, grad.get(cls), EPS);
          arg.adjust(cls, -EPS);
        }
      }
    }
  }
}
