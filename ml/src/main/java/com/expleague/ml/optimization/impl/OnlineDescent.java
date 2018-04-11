package com.expleague.ml.optimization.impl;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.ReguralizerFunc;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.StochasticGradientDescent;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;
import java.util.stream.IntStream;

import static java.lang.Math.log;

public class OnlineDescent implements Optimize<FuncEnsemble<? extends FuncC1>> {
  private final double step;
  private final FastRandom random;

  public OnlineDescent(final double step, final FastRandom random) {
    this.step = step;
    this.random = random;
  }

  private long time;
  @Override
  public Vec optimize(final FuncEnsemble<? extends FuncC1> ensemble, ReguralizerFunc reg, final Vec x0) {
    time = System.nanoTime();
    Vec cursor = VecTools.copy(x0);
    TIntArrayList taken = new TIntArrayList();
    List<Vec> lStat = new ArrayList<>();
    final Vec L = new ArrayVec(cursor.dim());
//    VecTools.fill(L, step / MathTools.EPSILON);

    final TDoubleArrayList lambdas = new TDoubleArrayList();
    final TDoubleArrayList gradModules = new TDoubleArrayList();
    final List<Vec> grads = new ArrayList<>();
    final Vec totalGrad = new ArrayVec(cursor.dim());

    Vec prev = VecTools.copy(cursor);
    for (int t = 0; t < ensemble.size(); t++) {
      int nextTaken = 0;
      {
        int nextIdx = random.nextInt(ensemble.size() - taken.size());
        for (int j = 0, takenIdx = 0; j < ensemble.size(); j++) {
          if (taken.size() > j && taken.get(takenIdx) == j) {
            takenIdx++;
          }
          else if (--nextIdx == 0) {
            taken.add(nextTaken = j);
            break;
          }
        }
      }
      final int samples = (int)Math.sqrt(t);
      final Vec grad = new ArrayVec(cursor.dim());
      final Vec next = new ArrayVec(cursor.dim());
      final Vec sampledGrad = new ArrayVec(cursor.dim());
      final FuncC1 model = ensemble.models[nextTaken];
      double v0 = model.value(cursor);


      int iter = 0;
      for (; iter < 100; iter++) {
        final Vec lambdasVec = new ArrayVec(lambdas.toArray());
        final int statSize = lStat.size();
        VecTools.fill(sampledGrad, 0);
        model.gradientTo(cursor, grad);
        VecTools.incscale(sampledGrad, grad, 1. / (samples + 1));
        IntStream.range(0, samples).parallel().forEach(s -> {
//          int sample = random.nextSimple(lambdasVec);
          int sample = random.nextInt(taken.size() - 1);

          Vec g = ensemble.models[taken.get(sample)].gradient(cursor);

          synchronized (sampledGrad) {
            VecTools.incscale(sampledGrad, g, 1. / (samples + 1));
            VecTools.incscale(totalGrad, grads.get(sample), -1);
            VecTools.append(totalGrad, g);
            gradModules.set(sample, VecTools.norm(g));
            grads.set(sample, VecTools.copySparse(g));
            lambdas.set(sample, Math.abs(VecTools.multiply(g, L) / VecTools.sum(g)));
          }
        });
        {
          for (int i = 0; i < sampledGrad.dim(); i++) {
            L.set(i, Math.max(L.get(i), Math.abs(sampledGrad.get(i))));
          }
          lStat.add(VecTools.copy(sampledGrad));
          if (statSize > 20000) {
            lStat = new ArrayList<>(lStat.subList(statSize - 10000, statSize));
            VecTools.fill(L, MathTools.EPSILON);
            lStat.forEach(v -> {
              for (int i = 0; i < v.dim(); i++) {
                L.set(i, Math.max(L.get(i), Math.abs(v.get(i))));
              }
            });
          }
        }
        double gradNorm = Math.sqrt(VecTools.sum2(sampledGrad) / sampledGrad.dim());
        VecTools.assign(next, cursor);
        for (int i = 0; i < cursor.dim(); i++) {
          double L_i = L.get(i);
          sampledGrad.set(i, step * sampledGrad.get(i) / (L_i > 0 ? L_i : MathTools.EPSILON));
        }
        VecTools.incscale(next, sampledGrad, -1);
        reg.project(next);
        if (gradNorm < 1e-3) {
          break;
        }
        VecTools.assign(cursor, next);
      }
      gradModules.add(VecTools.norm(grad));
      grads.add(VecTools.copySparse(grad));
      lambdas.add(Math.abs(VecTools.multiply(grad, L) / VecTools.sum(grad)));
      VecTools.append(totalGrad, grad);

      if (taken.size() % 100 == 0) {
        System.out.println("taken: " + taken.size() + " diff: " + VecTools.norm(VecTools.subtract(cursor, prev)) + " v_x^T= " + (model).value(cursor) + " V+R= " + (ensemble.value(cursor) + reg.value(cursor)) + " V= " + ensemble.value(cursor) + " iter: " + iter + " v_x^0= " + v0);
        System.out.println(cursor);
        prev = VecTools.copy(cursor);
      }
    }
    return cursor;
  }
}
