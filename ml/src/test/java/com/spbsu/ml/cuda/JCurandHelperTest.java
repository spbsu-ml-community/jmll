package com.spbsu.ml.cuda;

import org.junit.Ignore;
import org.junit.Test;

import com.xeiam.xchart.Histogram;
import jcuda.jcurand.curandGenerator;
import org.junit.Assert;

import java.util.*;

/**
 * Project jmll
 *
 * @author Ksen
 */
@Ignore
public class JCurandHelperTest extends Assert {

  @Test
  public void testGetDefaultFloatUniform() throws Exception {
    final curandGenerator generator = JCurandHelper.createDefault();
    final float[] floats = JCurandHelper.generateUniformHost(10_000, generator);
    JCurandHelper.destroyGenerator(generator);

    final Histogram histogram = new Histogram(transform(floats), 20);
    assertTrue(histogram.getMax() - histogram.getMin() < 150);
  }

  private Collection<Float> transform(final float[] array) {
    final ArrayList<Float> dArray = new ArrayList<>(array.length);
    for (int i = 0; i < array.length; i++) {
      dArray.add(array[i]);
    }
    return dArray;
  }

}
