package com.expleague.exp;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.tools.DataTools;
import org.junit.Test;

import java.util.stream.IntStream;

public class DataToolsTest {
  @Test
  public void testKMeans() {
    FastRandom rng = new FastRandom();
    final Vec[] centroids = IntStream.range(0, 100)
        .mapToObj(idx -> new ArrayVec(100))
        .peek(v -> VecTools.fillGaussian(v, rng))
        .toArray(Vec[]::new);
    final Vec[] answer = DataTools.kMeans(centroids.length, IntStream.range(0, 1000000).mapToObj(i ->
        VecTools.append(VecTools.fillGaussian(new ArrayVec(centroids[0].dim()), rng), centroids[rng.nextInt(centroids.length)]))
    );

    for (int i = 0; i < answer.length; i++) {

    }
  }
}
