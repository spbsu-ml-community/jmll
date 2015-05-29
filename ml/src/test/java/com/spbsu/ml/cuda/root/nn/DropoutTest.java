package com.spbsu.ml.cuda.root.nn;

import org.junit.Ignore;
import org.junit.Test;

import com.spbsu.ml.cuda.JCurandHelper;
import com.spbsu.ml.cuda.data.impl.FloatVector;
import jcuda.jcurand.curandGenerator;
import org.junit.Assert;

import java.util.Random;

/**
 * Project jmll
 *
 * @author Ksen
 */
@Ignore
public class DropoutTest extends Assert {

  private static final int LENGTH = 10_000;

  private static final float DELTA = 1e-9f;

  private static final Random RANDOM = new Random();

  @Test
  public void testDropoutTrain() throws Exception {
    final float[] inputs = generateHostFloatData();
    final float dropoutFraction = RANDOM.nextFloat();

    final FloatVector input = new FloatVector(inputs);
    final FloatVector dropoutMask = new FloatVector(new float[inputs.length]);
    final FloatVector output = new FloatVector(new float[inputs.length]);
    final curandGenerator generator = JCurandHelper.createDefault();

    Dropout.dropoutTrain(input, dropoutMask, output, generator, dropoutFraction);

    final float[] actual = new float[input.length];
    final float[] maskData = dropoutMask.get();
    for (int i = 0; i < actual.length; i++) {
      actual[i] = maskData[i] * inputs[i];
    }

    assertArrayEquals(actual, output.get(), DELTA);

    input.destroy();
    dropoutMask.destroy();
    output.destroy();
    JCurandHelper.destroyGenerator(generator);
  }

  @Test
  public void testDropoutTest() throws Exception {
    final float[] inputs = generateHostFloatData();
    final float dropoutFraction = RANDOM.nextFloat();

    final FloatVector input = new FloatVector(inputs);
    final FloatVector output = new FloatVector(new float[inputs.length]);

    Dropout.dropoutTest(input, output, dropoutFraction);

    final float[] actual = new float[input.length];
    for (int i = 0; i < actual.length; i++) {
      actual[i] = inputs[i] * (1.f - dropoutFraction);
    }

    assertArrayEquals(actual, output.get(), DELTA);

    input.destroy();
    output.destroy();
  }

  private float[] generateHostFloatData() {
    final float[] hostData = new float[LENGTH];
    for (int i = 0; i < hostData.length; i++) {
      hostData[i] = RANDOM.nextFloat();
    }
    return hostData;
  }

}
