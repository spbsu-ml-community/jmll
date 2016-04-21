package com.spbsu.exp;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import org.apache.commons.math3.util.FastMath;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;

/**
 * User: Noxoomo
 * Date: 21.04.16
 * Time: 23:27
 */
public class FastMathBenchmark {
  static FastRandom random = new FastRandom(0);
  static int dim = 10000;
  static Vec target = randomVec(dim);

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public double becnhmarkFastMathLog() {
    double sum = 0;
    double adjust = random.nextDouble();
    for (int i = 0; i < target.dim(); ++i) {
      sum += FastMath.log(0.5 * (target.get(i) + adjust));
    }
    return sum / target.dim();
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public double becnhmarkLog() {
    double sum = 0;
    double adjust = random.nextDouble();
    for (int i = 0; i < target.dim(); ++i) {
      sum += Math.log(0.5 * (target.get(i) + adjust));
    }
    return sum / target.dim();
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public double becnhmarkFastMathExp() {
    double sum = 0;
    double adjust = random.nextDouble();
    for (int i = 0; i < target.dim(); ++i) {
      sum += FastMath.exp(0.5 * (target.get(i) + adjust));
    }
    return sum / target.dim();
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  public double becnhmarkExp() {
    double sum = 0;
    double adjust = random.nextDouble();
    for (int i = 0; i < target.dim(); ++i) {
      sum += Math.exp(0.5 * (target.get(i) + adjust));
    }
    return sum / target.dim();
  }


  public static void main(String[] args) throws RunnerException {
    Options opt = new OptionsBuilder()
            .include(FastMathBenchmark.class.getSimpleName())
            .warmupIterations(50)
            .measurementIterations(100)
            .forks(1)
            .build();

    new Runner(opt).run();
  }

  static Vec randomVec(int dim) {
    Vec vec = new ArrayVec(dim);
    for (int i = 0; i < vec.dim(); ++i)
      vec.set(i, random.nextDouble() * 100000);
    return vec;
  }


}
