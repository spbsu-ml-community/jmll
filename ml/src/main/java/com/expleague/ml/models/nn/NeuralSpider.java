package com.expleague.ml.models.nn;

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
  public interface ForwardNode {
    double apply(Vec state, Vec betta, int nodeIdx);
    double activate(double value);
    double grad(double value);
    int start(int nodeIdx);
    int end(int nodeIdx);
  }

  public interface BackwardNode {
    double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int nodeIdx);
    int start(int nodeIdx);
    int end(int nodeIdx);
  }

  private final ThreadLocalArrayVec stateCache = new ThreadLocalArrayVec();
  private final ThreadLocalArrayVec gradientACache = new ThreadLocalArrayVec();
  private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

  public Vec compute(final NetworkBuilder<In>.Network network, In argument, Vec weights) {
    final Vec state = stateCache.get(network.stateDim());
    network.setInput(argument, state);

    Seq<ForwardNode> nodes = network.forwardFlow();

    produceState(nodes, weights, state);
    return network.outputFrom(state);
  }

  public Vec parametersGradient(final NetworkBuilder<In>.Network network, In argument,
                                TransC1 target, Vec weights, Vec gradWeight) {
    final Vec state = stateCache.get(network.stateDim());
    final Vec gradAct = gradientACache.get(network.stateDim());
    network.setInput(argument, state);

    final Seq<ForwardNode> nodes = network.forwardFlow();
    final Seq<BackwardNode> backwardNodes = network.backwardFlow();
    final Seq<BackwardNode> weightNodes = network.gradientFlow();

    final Vec gradState = gradientSCache.get(backwardNodes.length() + network.ydim());

    produceStateWithGrad(nodes, weights, state, gradAct);
    final Vec output = network.outputFrom(state);

    target.gradientTo(output, gradState.sub(gradState.length() - network.ydim(), network.ydim()));

    for (int nodeIdx = backwardNodes.length() - 1; nodeIdx >= 0; nodeIdx--) {
      final BackwardNode node = backwardNodes.at(nodeIdx);
      final double dTds_i = node.apply(state, gradState, gradAct, weights, nodeIdx);
      gradState.set(nodeIdx, dTds_i);
    }

    for (int nodeIdx = weightNodes.length() - 1; nodeIdx >= 0; nodeIdx--) {
      final BackwardNode node = weightNodes.at(nodeIdx);
      final double dTds_i = node.apply(state, gradState, gradAct, gradWeight, nodeIdx);
      gradWeight.set(nodeIdx, dTds_i);
    }

    return gradWeight;
  }

  private void produceState(Seq<ForwardNode> nodes, Vec weights, Vec state) {
    final int parallelism = ForkJoinPool.getCommonPoolParallelism();
    final int[] cursor = new int[parallelism + 1];
    final CountDownLatch latch = new CountDownLatch(parallelism);
    final int steps = (nodes.length() + parallelism - 1) / parallelism;
    for (int t = 0; t < parallelism; t++) {
      final int thread = t;
      ForkJoinPool.commonPool().execute(() ->
      {
        int i = 0;
        while(i < steps) {
          final int nodeIdx = thread + i * parallelism;
          if (nodeIdx >= state.length())
            break;

          final ForwardNode at = nodes.at(nodeIdx);
          int end = at.end(nodeIdx);
          if (end < cursor[0] + 1) {
            final double value = at.apply(state, weights, nodeIdx);
            state.set(nodeIdx, at.activate(value));
            cursor[thread + 1] = nodeIdx;
            i++;
          }
          else {
            cursor[0] = IntStream.range(1, cursor.length).map(idx -> cursor[idx])
                .sorted().reduce((a, b) -> a + 1 <= b ? a + 1 : a).orElse(0);
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

  private void produceStateWithGrad(Seq<ForwardNode> nodes, Vec weights, Vec state, Vec gradAct) {
    final int parallelism = ForkJoinPool.getCommonPoolParallelism();
    final int[] cursor = new int[parallelism + 1];
    final CountDownLatch latch = new CountDownLatch(parallelism);
    final int steps = (nodes.length() + parallelism - 1) / parallelism;
    for (int t = 0; t < parallelism; t++) {
      final int thread = t;
      ForkJoinPool.commonPool().execute(() ->
          {
            int i = 0;
            while(i < steps) {
              final int nodeIdx = thread + i * parallelism;
              if (nodeIdx >= state.length())
                break;

              final ForwardNode at = nodes.at(nodeIdx);
              int end = at.end(nodeIdx);
              if (end < cursor[0] + 1) {
                final double value = at.apply(state, weights, nodeIdx);
                state.set(nodeIdx, at.activate(value));
                gradAct.set(nodeIdx, at.grad(value));
                cursor[thread + 1] = nodeIdx;
                i++;
              }
              else {
                cursor[0] = IntStream.range(1, cursor.length).map(idx -> cursor[idx])
                    .sorted().reduce((a, b) -> a + 1 <= b ? a + 1 : a).orElse(0);
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
