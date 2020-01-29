package com.expleague.linear;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.ml.GridTest;
import com.expleague.ml.data.tools.FeaturesTxtPool;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Linear;
import com.expleague.ml.loss.L2;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.ml.methods.ElasticNetMethod;

import static com.expleague.commons.math.MathTools.sqr;
import static com.expleague.commons.math.vectors.VecTools.copy;

public class ElasticNetTest extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testElasticNetBenchmark() {
    //
    final int N = 20000;
    final int TestN = 20000;
    final int p = 2000;
    Vec beta = new ArrayVec(p);
    for (int i = 0; i < p; ++i) {
      beta.set(i, rng.nextGaussian());
    }
    Mx learn = new VecBasedMx(N, p);
    Mx test = new VecBasedMx(TestN, p);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < p; ++j) {
        learn.set(i, j, rng.nextDouble());
      }
    }
    for (int i = 0; i < TestN; ++i) {
      for (int j = 0; j < p; ++j) {
        test.set(i, j, rng.nextDouble());
      }
    }
    Vec realTarget = MxTools.multiply(learn, beta);
    Vec testTarget = MxTools.multiply(test, beta);
    Vec target = copy(realTarget);
    for (int i=0; i < target.dim();++i) {
      target.adjust(i, rng.nextGaussian()*0.005);
    }
    Pool pool = new FeaturesTxtPool(new ArraySeq<>(new QURLItem[target.length()]), learn, target);

    final L2 loss = (L2) pool.target(L2.class);
    long start_time = System.currentTimeMillis();


    double lambda = 1;
    Linear result;
    final ElasticNetMethod.ElasticNetCache cache = new ElasticNetMethod.ElasticNetCache(pool.vecData().data(), loss.target,0.95, lambda);
    final ElasticNetMethod net = new ElasticNetMethod(1e-2f, 0.95, 0);
    while (lambda > 1e-9) {
      cache.setLambda(lambda);
      result = net.fit(cache);
      System.out.println("Current lambda " + lambda);
      System.out.println("Current Fit time " + (System.currentTimeMillis() - start_time));
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
      System.out.println();
      lambda *= 0.9;
    }



    System.out.println("Current lambda " + 0);
    final ElasticNetMethod unregNet = new ElasticNetMethod(1e-2f, 0.95, 0);
    result = (Linear) unregNet.fit(pool.vecData(), loss);
    System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
    System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
    System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
    System.out.println();

    {
      System.out.println("Classic linear regression");
      final Mx trLearn = MxTools.transpose(learn);
      Vec classic = MxTools.multiply(MxTools.inverse(MxTools.multiply(trLearn, learn)), MxTools.multiply(trLearn,target));
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, classic), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn,  classic), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test,  classic), testTarget)) / testTarget.dim());
    }
//      System.out.println("Fit weights " + result.weights);
//      System.out.println("Real weights " + beta);
  }

  public void testElasticNet() {
    {
      final ElasticNetMethod net = new ElasticNetMethod(1e-2f, 0.5, 0);
      final int N = 100;
      final int p = 100;
      Vec beta = new ArrayVec(p);
      for (int i = 0; i < p; ++i) {
        beta.set(i, rng.nextDouble());
      }
      Mx learn = new VecBasedMx(N, p);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j)
          learn.set(i, j, rng.nextDouble());
      }
      Vec target = MxTools.multiply(learn, beta);
      Pool pool = new FeaturesTxtPool(new ArraySeq<>(new QURLItem[target.length()]), learn, target);
      final L2 loss = (L2) pool.target(L2.class);
      Linear result = (Linear) net.fit(pool.vecData(), loss);
      assertTrue(VecTools.distance(MxTools.multiply(learn, result.weights), target) < 1e-4f);
      assertTrue(VecTools.distance(beta, result.weights) < 1e-1f);
    }

    //
    {
      final ElasticNetMethod net = new ElasticNetMethod(1e-2f, 0.0, 0.0007);
      final int N = 1000;
      final int TestN = 20000;
      final int p = 100;
      Vec beta = new ArrayVec(p);
      for (int i = 0; i < p; ++i) {
        beta.set(i, rng.nextGaussian());
      }
      Mx learn = new VecBasedMx(N, p);
      Mx test = new VecBasedMx(TestN, p);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j) {
          learn.set(i, j, rng.nextDouble());
        }
      }
      for (int i = 0; i < TestN; ++i) {
        for (int j = 0; j < p; ++j) {
          test.set(i, j, rng.nextDouble());
        }
      }
      Vec realTarget = MxTools.multiply(learn, beta);
      Vec testTarget = MxTools.multiply(test, beta);
      Vec target = copy(realTarget);
      for (int i=0; i < target.dim();++i) {
        target.adjust(i, rng.nextGaussian()*0.001);
      }
      Pool pool = new FeaturesTxtPool(new ArraySeq<>(new QURLItem[target.length()]), learn, target);
      final L2 loss = (L2) pool.target(L2.class);
      Linear result = (Linear) net.fit(pool.vecData(), loss);
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
//      System.out.println("Fit weights " + result.weights);
//      System.out.println("Real weights " + beta);
    }


    //check shrinkage
    {
      final int N = 100;
      final int NTest = 1000;
      final int p = 500;
      Vec beta = new ArrayVec(p);
      for (int i = 0; i < p; ++i) {
        if (i % 17 == 0) {
          beta.set(i, rng.nextGaussian());
        } else {
          beta.set(i,0);
        }
      }
      Mx learn = new VecBasedMx(N, p);
      Mx test = new VecBasedMx(NTest, p);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j) {
          learn.set(i, j, rng.nextDouble());

        }
      }
      for (int i = 0; i < NTest; ++i) {
        for (int j = 0; j < p; ++j) {
          test.set(i, j, rng.nextDouble());
        }
      }
      Vec realTarget = MxTools.multiply(learn, beta);
      Vec testTarget = MxTools.multiply(test, beta);
      Vec target = copy(realTarget);
      for (int i=0; i < target.dim();++i) {
        target.adjust(i, rng.nextGaussian() * 0.05);
      }
      Pool pool = new FeaturesTxtPool(new ArraySeq<>(new QURLItem[target.length()]), learn, target);
      final L2 loss = (L2) pool.target(L2.class);

      double lambda = 0.01;
      Vec w = new ArrayVec(beta.dim());
      Linear result;
      while (lambda > 0.00000001) {
        final ElasticNetMethod net = new ElasticNetMethod(1e-5f, 0.95, lambda);
        result = (Linear) net.fit(pool.vecData(), loss,w);
        System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
        System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
        System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
        w = result.weights;
        lambda *= 0.75;
      }

      final ElasticNetMethod net = new ElasticNetMethod(1e-5f, 0.95, 0);
      result = (Linear) net.fit(pool.vecData(), loss);
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
      w = result.weights;
//      for (int i=0; i < beta.xdim();++i) {
//        if (beta.get(i) == 0)
//          assertTrue(Math. abs(result.weights.get(i)-0.0) < 1e-9);
//      }
//      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.xdim());
//      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.xdim());
//      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.xdim());
//      System.out.println("Fit weights " + result.weights);
//      System.out.println("Real weights " + beta);
    }
  }


  public void testElasticNetPath() {
    final int N = 250;
    final int NTest = 1000;
    final int p = 220;
    Vec beta = new ArrayVec(p);
    for (int i = 0; i < p; ++i) {
      if (i % 17 == 0) {
        beta.set(i, rng.nextGaussian());
      } else {
        beta.set(i,0);
      }
    }
    Mx learn = new VecBasedMx(N, p);
    Mx test = new VecBasedMx(NTest, p);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < p; ++j) {
        learn.set(i, j, rng.nextDouble());

      }
    }
    for (int i = 0; i < NTest; ++i) {
      for (int j = 0; j < p; ++j) {
        test.set(i, j, rng.nextDouble());
      }
    }
    Vec realTarget = MxTools.multiply(learn, beta);
    Vec testTarget = MxTools.multiply(test, beta);
    Vec target = copy(realTarget);
    for (int i=0; i < target.dim();++i) {
      target.adjust(i, rng.nextGaussian() * 0.05);
    }
    Pool pool = new FeaturesTxtPool(new ArraySeq<>(new QURLItem[target.length()]), learn, target);
    final L2 loss = (L2) pool.target(L2.class);

    double lambda = 1;
    Linear result;
    final ElasticNetMethod.ElasticNetCache cache = new ElasticNetMethod.ElasticNetCache(pool.vecData().data(), loss.target,0.99, lambda);
    final ElasticNetMethod net = new ElasticNetMethod(1e-5f, 0.99, 0);
    while (lambda > 1e-5) {
      cache.setLambda(lambda);
      result = net.fit(cache);
      System.out.println("Current lambda " + lambda);
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
      System.out.println();
      lambda *= 0.85;
    }

    System.out.println("Current lambda " + 0);
    final ElasticNetMethod unregNet = new ElasticNetMethod(1e-7f, 0.99, 0);
    result = (Linear) unregNet.fit(pool.vecData(), loss);
    System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
    System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
    System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
    System.out.println();

    {
      System.out.println("Classic linear regression");
      final Mx trLearn = MxTools.transpose(learn);
      Vec classic = MxTools.multiply(MxTools.inverse(MxTools.multiply(trLearn, learn)), MxTools.multiply(trLearn,target));
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, classic), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn,  classic), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test,  classic), testTarget)) / testTarget.dim());
    }
  }
}
