package com.expleague.ml.models.nn;

import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.ThreadTools;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static java.util.Arrays.stream;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 12:57
 */
public class NeuralSpider<In> {
  private final static int parallelism = ThreadTools.COMPUTE_UNITS;
  private ThreadPoolExecutor poolExecutor = ThreadTools.createBGExecutor("NeuralSpider calculators", parallelism);

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

    class Stub implements BackwardNode {
      @Override
      public double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int nodeIdx) {
        return 0;
      }

      @Override
      public int start(int nodeIdx) {
        return 0;
      }

      @Override
      public int end(int nodeIdx) {
        return 0;
      }
    }
  }

  private final ThreadLocalArrayVec stateCache = new ThreadLocalArrayVec();
  private final ThreadLocalArrayVec gradientACache = new ThreadLocalArrayVec();
  private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

  public synchronized Vec compute(final NetworkBuilder<In>.Network network, In argument, Vec weights) {
    final Vec state = stateCache.get(network.stateDim());
    network.setInput(argument, state);

    Seq<ForwardNode> nodes = network.forwardFlow();

    produceState(nodes, weights, state);
    return network.outputFrom(state);
  }

  public synchronized Vec parametersGradient(final NetworkBuilder<In>.Network network, In argument,
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

    {
      final CountDownLatch latch = new CountDownLatch(parallelism);
      final int steps = (backwardNodes.length() + parallelism - 1) / parallelism;
      final int[] cursor = new int[parallelism + 1];
      cursor[0] = backwardNodes.length();
      for (int i = 1; i < cursor.length; i++) {
        cursor[i] = backwardNodes.length();
      }

      for (int t = 0; t < parallelism; t++) {
        final int thread = t;
        this.poolExecutor.execute(() -> {
          int i = steps;
          int counter = 0;
          while (i >= 0) {
            final int nodeIdx = thread + i * parallelism;
            if (nodeIdx >= backwardNodes.length()) {
              i--;
              continue;
            }
            final BackwardNode node = backwardNodes.at(nodeIdx);
            final int start = node.start(nodeIdx);
            cursor[thread + 1] = start;

            if (start >= cursor[0]) {
              final double dTds_i = node.apply(state, gradState, gradAct, weights, nodeIdx);
              gradState.set(nodeIdx, dTds_i);
              i--;
            } else {
              int max = Integer.MIN_VALUE;
              for (int k = 1; k < cursor.length; k++) {
                max = Math.max(cursor[k], max);
              }
              //noinspection StatementWithEmptyBody
              while (ArrayTools.indexOf(--max, cursor) > 0);
              cursor[0] = max + 1;
              if (start < cursor[0]) {
                if (counter++ > 1000000) {
                  counter = 0;
                  Thread.yield();
                }
              }
              else counter = 0;
            }
          }
          cursor[thread + 1] = 0;
          latch.countDown();
        });
      }
      try {
        latch.await();
      }
      catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }

    {
      final CountDownLatch latch = new CountDownLatch(parallelism);
      final int steps = (weightNodes.length() + parallelism - 1) / parallelism;

      for (int t = 0; t < parallelism; t++) {
        final int thread = t;
        this.poolExecutor.execute(() -> {
          for (int i = steps; i >= 0; i--) {
            final int nodeIdx = thread + i * parallelism;
            if (nodeIdx >= weightNodes.length())
              continue;
            final BackwardNode node = weightNodes.at(nodeIdx);
            final double dTds_i = node.apply(state, gradState, gradAct, gradWeight, nodeIdx);
            gradWeight.set(nodeIdx, dTds_i);
          }
          latch.countDown();
        });
      }
      try {
        latch.await();
      }
      catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }

    return gradWeight;
  }


  private void produceState(Seq<ForwardNode> nodes, Vec weights, Vec state) {
    final int[] cursor = new int[parallelism + 1];
    final CountDownLatch latch = new CountDownLatch(parallelism);
    final int steps = (nodes.length() + parallelism - 1) / parallelism;
    for (int t = 0; t < parallelism; t++) {
      final int thread = t;
      this.poolExecutor.execute(() ->
      {
        int i = 0;
        while(i < steps) {
          final int nodeIdx = thread + i * parallelism;
          if (nodeIdx >= state.length())
            break;

          final ForwardNode at = nodes.at(nodeIdx);
          final int end = at.end(nodeIdx);
          cursor[thread + 1] = end;

          if (end <= cursor[0]) {
            final double value = at.apply(state, weights, nodeIdx);
            state.set(nodeIdx, at.activate(value));
            i++;
          }
          else {
            cursor[0] = stream(cursor, 1, cursor.length)
                .sorted().min().getAsInt();
            if (end > cursor[0]) {
              Thread.yield();
            }
          }
        }
        cursor[thread + 1] = nodes.length();
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
    final int[] cursor = new int[parallelism + 1];
    final CountDownLatch latch = new CountDownLatch(parallelism);
    final int steps = (nodes.length() + parallelism - 1) / parallelism;
    for (int t = 0; t < parallelism; t++) {
      final int thread = t;
      this.poolExecutor.execute(() ->
          {
            int i = 0;
            int counter = 0;
            while(i < steps) {
              final int nodeIdx = thread + i * parallelism;
              if (nodeIdx >= state.length())
                break;

              final ForwardNode at = nodes.at(nodeIdx);
              int end = at.end(nodeIdx);
              cursor[thread + 1] = end;

              if (end <= cursor[0]) {
                final double value = at.apply(state, weights, nodeIdx);
                state.set(nodeIdx, at.activate(value));
                gradAct.set(nodeIdx, at.grad(value));
                i++;
              }
              else {
                int min = Integer.MAX_VALUE;
                for (int k = 1; k < cursor.length; k++) {
                  min = Math.min(cursor[k], min);
                }
                //noinspection StatementWithEmptyBody
                while (ArrayTools.indexOf(++min, cursor) > 0);
                cursor[0] = min - 1;
                if (end > cursor[0]) {
                  if (counter++ > 1000000) {
                    counter = 0;
                    Thread.yield();
                  }
                }
                else counter = 0;
              }
            }
            cursor[thread + 1] = nodes.length();
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
