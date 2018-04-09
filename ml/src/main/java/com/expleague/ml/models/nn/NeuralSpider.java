package com.expleague.ml.models.nn;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.expleague.commons.seq.Seq;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 12:57
 */
public class NeuralSpider<In> {
  public interface NodeCalcer {
    double apply(Vec state, Vec betta, int nodeIdx);
    int start(int nodeIdx);
    int end(int nodeIdx);
    void gradByStateTo(Vec state, Vec betta, int nodeIdx, double prevGrad, Vec gradState);
    void gradByParametersTo(Vec state, Vec betta, int nodeIdx, double prevGrad, Vec gradWeight);
  }

  private final ThreadLocalArrayVec stateCache = new ThreadLocalArrayVec();
  private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

  public Vec compute(final NetworkBuilder<In>.Network network, In argument, Vec weights) {
    final Vec state = stateCache.get(network.stateDim());
    network.setInput(argument, state);

    Seq<NodeCalcer> calcers = network.materialize();

    produceState(calcers, weights, state);
    return network.outputFrom(state);
  }

  public Vec parametersGradient(final NetworkBuilder<Object>.Network network, In argument,
                                TransC1 target, Vec weights, Vec gradWeight) {
    final Vec state = stateCache.get(network.stateDim());
    network.setInput(argument, state);
    final Seq<NodeCalcer> calcers = network.materialize();

    final Vec gradState = gradientSCache.get(network.stateDim());

    produceState(calcers, weights, state);
    final Vec output = network.outputFrom(state);

    target.gradientTo(output, gradState.sub(gradState.length() - network.ydim(), network.ydim()));
    for (int nodeIdx = calcers.length() - 1; nodeIdx > 0; nodeIdx--) {
      final NodeCalcer node = calcers.at(nodeIdx);

      final double dTds_i = gradState.get(nodeIdx);
      if (Math.abs(dTds_i) < MathTools.EPSILON || Math.abs(state.get(nodeIdx)) < MathTools.EPSILON)
        continue;

      node.gradByStateTo(state, weights, nodeIdx, dTds_i, gradState);
      node.gradByParametersTo(state, weights, nodeIdx, dTds_i, gradWeight);
    }

    return gradWeight;
  }

  long counter;
  protected void produceState(Seq<NodeCalcer> calcers, Vec weights, Vec state) {
    counter = 0;
    final int parallelism = ForkJoinPool.getCommonPoolParallelism();
    final int[] cursor = new int[parallelism + 1];
    final CountDownLatch latch = new CountDownLatch(parallelism);
    final int steps = (calcers.length() + parallelism - 1) / parallelism;
    for (int t = 0; t < parallelism; t++) {
      final int thread = t;
      ForkJoinPool.commonPool().execute(() ->
      {
        int i = 0;
        while(i < steps) {
          final int nodeIdx = thread + i * parallelism;
          if (nodeIdx >= state.length())
            break;

          final NodeCalcer at = calcers.at(nodeIdx);
          int end = at.end(nodeIdx);
          if (end <= cursor[0] + 1) {
            state.set(nodeIdx, at.apply(state, weights, nodeIdx));
            cursor[thread + 1] = nodeIdx;
            i++;
          }
          else {
            cursor[0] = IntStream.range(1, cursor.length).map(idx -> cursor[idx])
                .sorted().reduce((a, b) -> a + 1 <= b ? a + 1 : a).orElse(0);
            counter++;
            if (cursor[0] < end) {
              Thread.yield();
            }
          }
        }
        latch.countDown();
      }
      );
    }
    try {
      latch.await();
    }
    catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }
}
