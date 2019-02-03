package com.expleague.cuda.root.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.cuda.JCudaHelper;
import com.expleague.cuda.data.GPUVec;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.seq.PNFA;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by hrundelb on 06.09.17.
 */
public class PNFAonGPUTest {

  private final static float DELTA = 1e-3f;

  @BeforeClass
  public static void initCuda() {
    Assume.assumeNoException(JCudaHelper.checkInstance());
  }

  @Test
  public void testLoss() {
    FastRandom random = new FastRandom();
    int stateCount = 17;
    int alphabetSize = 5;
    double y = random.nextDouble();
    double w = random.nextDouble();
    double[] hostArray = new double[stateCount * (stateCount - 1) * alphabetSize + stateCount];
    for (int i = 0; i < hostArray.length; i++) {
      hostArray[i] = random.nextDouble();
    }
    int[] ints = new int[alphabetSize];
    for (int i = 0; i < ints.length; i++) {
      ints[i] = random.nextInt(alphabetSize);
    }
    IntSeq seq = new IntSeq(ints);

    PNFAonGPU<WeightedL2> pnfAonGPU = new PNFAonGPU<>(stateCount, alphabetSize, random, null);
    PNFAonGPU.PNFAPointLossFunc func = pnfAonGPU.new PNFAPointLossFunc(seq, y, w);

    PNFA<WeightedL2> pnfa = new PNFA<>(stateCount, 1, alphabetSize, 0.01, 0, random, null,null, 1);
    PNFA.PNFAPointLossFunc pointLossFunc = pnfa.new PNFAPointLossFunc(seq, new ArrayVec(y), w);

    double v = pointLossFunc.value(new ArrayVec(hostArray));
    System.out.println(v);

    double value = func.value(new GPUVec(hostArray));
    System.out.println(value);

    assertEquals(value, v, DELTA);

    Vec gradient = pointLossFunc.gradient(new ArrayVec(hostArray));
    System.out.println(Arrays.toString(gradient.toArray()));

    Vec gradient1 = func.gradient(new GPUVec(hostArray));
    System.out.println(Arrays.toString(gradient1.toArray()));

    assertArrayEquals(gradient.toArray(), gradient1.toArray(), DELTA);
  }


  @Test
  public void test() {

  }

}
