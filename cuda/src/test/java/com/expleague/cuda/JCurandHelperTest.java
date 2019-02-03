package com.expleague.cuda;

import org.junit.*;

import com.xeiam.xchart.Histogram;
import jcuda.jcurand.curandGenerator;

import java.util.*;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class JCurandHelperTest extends Assert {

  @BeforeClass
  public static void initCuda() {
    Assume.assumeNoException(JCudaHelper.checkInstance());
  }

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
