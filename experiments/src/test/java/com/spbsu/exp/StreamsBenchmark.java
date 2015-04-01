package com.spbsu.exp;

/**
 * Created by noxoomo on 31/03/15.
 */

/*
To use: add to pom.xml, set java to 8 and uncomment
<dependency>
            <groupId>org.openjdk.jmh</groupId>
            <artifactId>jmh-core</artifactId>
            <version>${jmh.version}</version>
        </dependency>
        <dependency>
            <groupId>org.openjdk.jmh</groupId>
            <artifactId>jmh-generator-annprocess</artifactId>
            <version>${jmh.version}</version>
            <scope>provided</scope>
        </dependency>
 */
//
//public class StreamsBenchmark {
//  static FastRandom random = new FastRandom(0);
//  static int dim = 10000;
//
////  @Benchmark
////  public Vec becnhmarkL2Gradient() {
////    Vec target = randomVec(dim);
////    Vec x = randomVec(dim);
////    Vec b = sequentialL2Gradient(target, x);
////    return b;
////  }
////
////  @Benchmark
////  public Vec becnhmarkL2ParallelGradient() {
////    Vec target = randomVec(dim);
////    Vec x = randomVec(dim);
////    Vec b = parallelL2Gradient(target, x);
////    return b;
////  }
//
//  @Benchmark
//  @BenchmarkMode(Mode.AverageTime)
//  @OutputTimeUnit(TimeUnit.MICROSECONDS)
//  public Vec becnhmarkLLGradient() {
//    Vec target = randomBinVec(dim);
//    Vec x = randomVec(dim);
//    Vec b = sequentialLLGradient(target, x);
//    return b;
//  }
//
//  @Benchmark
//  @BenchmarkMode(Mode.AverageTime)
//  @OutputTimeUnit(TimeUnit.MICROSECONDS)
//  public Vec becnhmarkLLParallelGradient() {
//    Vec target = randomBinVec(dim);
//    Vec x = randomVec(dim);
//    Vec b = parallelLLGradient(target, x);
//    return b;
//  }
//
//  public static void main(String[] args) throws RunnerException {
//    Options opt = new OptionsBuilder()
//            .include(StreamsBenchmark.class.getSimpleName())
//            .warmupIterations(3)
//            .measurementIterations(5)
//            .forks(1)
//            .build();
//
//    new Runner(opt).run();
////    System.out.println("LL");
////    testLLGradient();
////    System.out.println("L2");
////    testL2Gradient();
//  }
//
//  static Vec randomVec(int dim) {
//    Vec vec = new ArrayVec(dim);
//    for (int i = 0; i < vec.dim(); ++i)
//      vec.set(i, random.nextDouble());
//    return vec;
//  }
//
//  static Vec randomBinVec(int dim) {
//    Vec vec = new ArrayVec(dim);
//    for (int i = 0; i < vec.dim(); ++i)
//      vec.set(i, random.nextDouble() > 0.5 ? 1 : -1);
//    return vec;
//  }
//
//
//  static int warmUp = 1000;
//  static int tries = 100000;
//
//  static Vec parallelL2Gradient(final Vec target, final Vec x) {
//    final Vec result = new ArrayVec(x.dim());
//    IntStream.range(0, x.dim()).parallel().forEach(i -> result.set(i, 2 * (x.get(i) - target.get(i))));
//    return result;
//  }
//
//  static Vec sequentialL2Gradient(final Vec target, final Vec x) {
//    final Vec result = copy(x);
//    scale(result, -1);
//    append(result, target);
//    scale(result, -2);
//    return result;
//  }
//
//  static Vec sequentialLLGradient(final Vec target, final Vec x) {
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
//  static Vec parallelLLGradient(final Vec target, final Vec x) {
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
//  static public void testL2Gradient() {
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
//        if (distance(a, b) > 1e-9) {
//          System.err.println("error");
//        }
//      }
//      System.out.println("Dim: " + dim + "\nWorking time for parallel gradient " + (parallelTime) / 1000);
//      System.out.println("Working time for seq gradient " + (sequatialTime) / 1000);
//    }
//  }
//
//
//  static public void testLLGradient() {
//    double tmp = 0;
//    for (int i = 0; i < warmUp; ++i) {
//      Vec tgt = randomVec(10000);
//      Vec x = randomVec(10000);
//      Vec a = parallelLLGradient(tgt, x);
//      Vec b = sequentialLLGradient(tgt, x);
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
//          b = sequentialLLGradient(target, x);
//          long endTime = System.currentTimeMillis();
//          sequentialTime += endTime - startTime;
//        }
//        if (distance(a, b) > 1e-9) {
//          System.err.println("error");
//        }
//      }
//      System.out.println("Dim: " + dim + "\nWorking time for parallel gradient " + (parallelTime) / 1000);
//      System.out.println("Working time for seq gradient " + (sequentialTime) / 1000);
//    }
//  }
//}
