package com.spbsu.exp;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import junit.framework.TestCase;

//import java.util.stream.IntStream;

import static com.spbsu.commons.math.vectors.VecTools.*;
import static java.lang.Math.exp;

/**
 * Created by noxoomo on 31/03/15.
 */
public class Java8ParallelStreamsTest extends TestCase {
//  FastRandom random = new FastRandom(0);
//
//  private Vec randomVec(int dim) {
//    Vec vec = new ArrayVec(dim);
//    for (int i = 0; i < vec.dim(); ++i)
//      vec.set(i, random.nextDouble());
//    return vec;
//  }
//
//  private Vec randomBinVec(int dim) {
//    Vec vec = new ArrayVec(dim);
//    for (int i = 0; i < vec.dim(); ++i)
//      vec.set(i, random.nextDouble() > 0.5 ? 1 : -1);
//    return vec;
//  }
//
//
//  int warmUp = 1000;
//  int tries = 100000;
//
//  Vec parallelL2Gradient(final Vec target, final Vec x) {
//    final Vec result = new ArrayVec(x.dim());
//    IntStream.range(0, x.dim()).parallel().forEach(i -> result.set(i, 2 * (x.get(i) - target.get(i))));
//    return result;
//  }
//
//  Vec sequentialL2Gradient(final Vec target, final Vec x) {
//    final Vec result = copy(x);
//    scale(result, -1);
//    append(result, target);
//    scale(result, -2);
//    return result;
//  }
//
//  Vec sequentiallLLGradient(final Vec target, final Vec x) {
//    final Vec result = new ArrayVec(x.dim());
//    for (int i = 0; i < x.dim(); i++) {
//      final double expX = exp(x.get(i));
//      final double pX = expX / (1 + expX);
//      if (target.get(i) > 0) // positive example
//        result.set(i, pX - 1);
//      else // negative
//        result.set(i, pX);
//    }
//    return result;
//  }
//
//  Vec parallelLLGradient(final Vec target, final Vec x) {
//    final Vec result = new ArrayVec(x.dim());
//    IntStream.range(0, x.dim()).parallel().forEach(i -> {
//      final double expX = exp(x.get(i));
//      final double pX = expX / (1 + expX);
//      result.set(i, target.get(i) > 0 ? pX - 1 : pX);
//    });
//    return result;
//  }
//
//
//  public void testL2Gradient() {
//    double tmp = 0;
//    for (int i = 0; i < warmUp; ++i) {
//      Vec tgt = randomVec(10000);
//      Vec x = randomVec(10000);
//      Vec a = parallelL2Gradient(tgt, x);
//      Vec b = sequentialL2Gradient(tgt, x);
//      tmp += distance(a, b);
//    }
//    for (int dim = 1000; dim < 1000000; dim *= 10) {
//      long parallelTime = 0;
//      long sequatialTime = 0;
//      for (int i = 0; i < tries; ++i) {
//        Vec target = randomVec(dim);
//        Vec x = randomVec(dim);
//        Vec a;
//        {
//          long startTime = System.currentTimeMillis();
//          a = parallelL2Gradient(target, x);
//          long endTime = System.currentTimeMillis();
//          parallelTime += endTime - startTime;
//        }
//        Vec b;
//        {
//          long startTime = System.currentTimeMillis();
//          b = sequentialL2Gradient(target, x);
//          long endTime = System.currentTimeMillis();
//          sequatialTime += endTime - startTime;
//        }
//        TestCase.assertTrue(distance(a, b) < 1e-9);
//      }
//      System.out.println("Dim: " + dim + "\nWorking time for parallel gradient " + (parallelTime) / 1000);
//      System.out.println("Working time for seq gradient " + (sequatialTime) / 1000);
//    }
//  }
//
//
//  public void testLLGradient() {
//    double tmp = 0;
//    for (int i = 0; i < warmUp; ++i) {
//      Vec tgt = randomVec(10000);
//      Vec x = randomVec(10000);
//      Vec a = parallelL2Gradient(tgt, x);
//      Vec b = sequentialL2Gradient(tgt, x);
//      tmp += distance(a, b);
//    }
//    for (int dim = 1000; dim < 1000000; dim *= 10) {
//      long parallelTime = 0;
//      long sequentialTime = 0;
//      for (int i = 0; i < tries; ++i) {
//        Vec target = randomBinVec(dim);
//        Vec x = randomVec(dim);
//        Vec a;
//        {
//          long startTime = System.currentTimeMillis();
//          a = parallelLLGradient(target, x);
//          long endTime = System.currentTimeMillis();
//          parallelTime += endTime - startTime;
//        }
//        Vec b;
//        {
//          long startTime = System.currentTimeMillis();
//          b = sequentiallLLGradient(target, x);
//          long endTime = System.currentTimeMillis();
//          sequentialTime += endTime - startTime;
//        }
//        TestCase.assertTrue(distance(a, b) < 1e-9);
//      }
//      System.out.println("Dim: " + dim + "\nWorking time for parallel gradient " + (parallelTime) / 1000);
//      System.out.println("Working time for seq gradient " + (sequentialTime) / 1000);
//    }
//  }
//
//
//  FastRandom rand1 = new FastRandom(0);
//  FastRandom rand2 = new FastRandom(0);
//
//  private int[] bootsrap(int dim) {
//    final int[] poissonWeights = new int[dim];
//    for (int i = 0; i < dim; i++) {
//      poissonWeights[i] = rand1.nextPoisson(1.);
//    }
//    return poissonWeights;
//  }
//
//  private int[] parallelBootstrap(int dim,final FastRandom[] rands) {
//    final int[] poissonWeights = new int[dim];
//    IntStream.range(0, poissonWeights.length).parallel().forEach(i -> poissonWeights[i] = rands[i].nextPoisson(1.0));
//    return poissonWeights;
//  }
//
//  public void testBootstrap() {
//    {
//      FastRandom[] rands = new FastRandom[10000];
//      for (int i=0; i < rands.length;++i)
//        rands[i] = new FastRandom(rand2.nextLong());
//      for (int i = 0; i < warmUp; ++i) {
//        int[] a = bootsrap(10000);
//        int[] b = parallelBootstrap(10000, rands);
//      }
//    }
//
//    for (int dim = 1000; dim < 1000000; dim *= 10) {
//      long parallelTime = 0;
//      long sequentialTime = 0;
//      FastRandom[] rands = new FastRandom[dim];
//      for (int i=0; i < rands.length;++i)
//        rands[i] = new FastRandom(rand2.nextLong());
//      for (int i = 0; i < tries; ++i) {
//        int[] a;
//        {
//          long startTime = System.currentTimeMillis();
//          a = parallelBootstrap(dim,rands);
//          long endTime = System.currentTimeMillis();
//          parallelTime += endTime - startTime;
//        }
//        int[] b;
//        {
//          long startTime = System.currentTimeMillis();
//          b = bootsrap(dim);
//          long endTime = System.currentTimeMillis();
//          sequentialTime += endTime - startTime;
//        }
//      }
//      System.out.println("Dim: " + dim + "\nWorking time for parallel bootstrap " + (parallelTime) / 1000);
//      System.out.println("Working time for seq bootstrap " + (sequentialTime) / 1000);
//    }
//  }
}
