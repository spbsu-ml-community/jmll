package com.expleague.exp;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.tools.DataTools;
import org.junit.Assert;
import org.junit.Test;

import java.util.stream.IntStream;
import java.util.stream.Stream;

public class DataToolsTest {
  @Test
  public void testKMeansSingleCentroid() {
    FastRandom rng = new FastRandom(100500);
    final Vec[] centroids = IntStream.range(0, 1)
        .mapToObj(idx -> new ArrayVec(10))
        .peek(v -> VecTools.fillGaussian(v, rng))
        .toArray(Vec[]::new);
    Vec sum = new ArrayVec(centroids[0].dim());
    int[] count = new int[]{0};
    final Vec[] answer = DataTools.kMeans(centroids.length, 1, rng, IntStream.range(0, 10000).mapToObj(i -> {
      final ArrayVec next = VecTools.append(VecTools.fillGaussian(new ArrayVec(centroids[0].dim()), rng), centroids[rng.nextInt(centroids.length)]);
      VecTools.append(sum, next);
      count[0]++;
      return next;
    }));

    VecTools.scale(sum, 1./count[0]);

    double totalDist = 0;
    for (int i = 0; i < answer.length; i++) {
      final Vec q = answer[i];
      final double minDist = Stream.of(centroids).mapToDouble(v -> VecTools.distance(q, v)).min().orElse(Double.POSITIVE_INFINITY);
      totalDist += minDist;
    }
    Assert.assertTrue("Mean residual: " + (totalDist / centroids.length) + " while mean diff is: " + VecTools.distance(centroids[0], sum), totalDist < centroids[0].dim() * 1e-2);
  }

  @Test
  public void testKMeansNoNNErrors() {
    FastRandom rng = new FastRandom(100500);
    final Vec[] centroids = IntStream.range(0, 2)
        .mapToObj(idx -> new ArrayVec(10))
        .peek(v -> VecTools.fillGaussian(v, rng))
        .peek(v -> VecTools.scale(v, 10))
        .toArray(Vec[]::new);
    final Vec[] answer = DataTools.kMeans(centroids.length, 2, rng, IntStream.range(0, 100000).mapToObj(i ->
        VecTools.append(VecTools.fillGaussian(new ArrayVec(centroids[0].dim()), rng), centroids[rng.nextInt(centroids.length)]))
    );

    double totalDist = 0;
    for (int i = 0; i < answer.length; i++) {
      final Vec q = answer[i];
      final double minDist = Stream.of(centroids).mapToDouble(v -> VecTools.distance(q, v)).min().orElse(Double.POSITIVE_INFINITY);
      totalDist += minDist;
    }
    Assert.assertTrue("Mean residual: " + (totalDist / centroids.length) + " while mean norm is: " + Stream.of(centroids).mapToDouble(VecTools::norm).average().orElse(Double.POSITIVE_INFINITY), totalDist / answer.length < centroids[0].dim() * 1e-2);
  }

  @Test
  public void testKMeansTen() {
    FastRandom rng = new FastRandom(100500);
    final Vec[] centroids = IntStream.range(0, 10)
        .mapToObj(idx -> new ArrayVec(10))
        .peek(v -> VecTools.fillGaussian(v, rng))
        .peek(v -> VecTools.scale(v, 10))
        .toArray(Vec[]::new);
    final Vec[] answer = DataTools.kMeans(centroids.length, 5, rng, IntStream.range(0, 100000).mapToObj(i ->
        VecTools.append(VecTools.fillGaussian(new ArrayVec(centroids[0].dim()), rng), centroids[rng.nextInt(centroids.length)]))
    );

    double totalDist = 0;
    for (int i = 0; i < answer.length; i++) {
      final Vec q = answer[i];
      final double minDist = Stream.of(centroids).mapToDouble(v -> VecTools.distance(q, v)).min().orElse(Double.POSITIVE_INFINITY);
      totalDist += minDist;
    }
    Assert.assertTrue("Mean residual: " + (totalDist / centroids.length) + " while mean norm is: " + Stream.of(centroids).mapToDouble(VecTools::norm).average().orElse(Double.POSITIVE_INFINITY), totalDist / answer.length < centroids[0].dim() * 1e-2);
  }

  @Test
  public void testKMeansHundred() {
    FastRandom rng = new FastRandom(100500);
    final Vec[] centroids = IntStream.range(0, 100)
        .mapToObj(idx -> new ArrayVec(10))
        .peek(v -> VecTools.fillGaussian(v, rng))
        .peek(v -> VecTools.scale(v, 10))
        .toArray(Vec[]::new);
    final Vec[] answer = DataTools.kMeans(centroids.length, 15, rng, IntStream.range(0, 100000).mapToObj(i ->
        VecTools.append(VecTools.fillGaussian(new ArrayVec(centroids[0].dim()), rng), centroids[rng.nextInt(centroids.length)]))
    );

    double totalDist = 0;
    for (int i = 0; i < answer.length; i++) {
      final Vec q = answer[i];
      final double minDist = Stream.of(centroids).mapToDouble(v -> VecTools.distance(q, v)).min().orElse(Double.POSITIVE_INFINITY);
      totalDist += minDist;
    }
    final double v = totalDist / centroids.length;
    Assert.assertTrue("Mean residual: " + v + " while mean norm is: " + Stream.of(centroids).mapToDouble(VecTools::norm).average().orElse(Double.POSITIVE_INFINITY), v < centroids[0].dim() * 5e-1);
  }
}
