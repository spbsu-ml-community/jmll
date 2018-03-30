package com.expleague.ml.models.nn;

import com.expleague.commons.math.TransC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.expleague.commons.seq.Seq;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 12:57
 */
public class NeuralSpider<In> {
  private final ThreadLocalArrayVec stateCache = new ThreadLocalArrayVec();
  private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

  public Vec compute(final NetworkBuilder<In>.Network network, In argument, Vec weights) {
    final Vec state = stateCache.get(network.xdim(argument));
    Seq<NodeCalcer> calcers = network.materialize(argument, state);
    produceState(calcers, weights, state);
    return network.outputFrom(state);
  }

  public Vec parametersGradient(In argument, TransC1 target, Vec allWeights, Vec to) {
    throw new NotImplementedException();
//    final Topology topology = topology(true);
//    final Vec dT = gradientSCache.get(topology.length());
//    final Vec state = stateCache.get(topology.length());
//    final Vec arg = new ArrayVec(argument.toArray());
//    for (int i = 0; i < arg.length(); i++) {
//      state.set(i, arg.get(i));
//    }
//    final Vec output = produceState(topology, allWeights, state);
//
//    target.gradientTo(output, dT.sub(dT.length() - topology.outputCount(), topology.outputCount()));
//    final Vec prevLayerGrad = new SparseVec(state.xdim());
//    final Vec wGrad = new SparseVec(allWeights.xdim());
//    for (int topologIdx = topology.length() - 1; topologIdx > 0; topologIdx--) {
//      final NodeCalcer nodeCalcer = topology.at(topologIdx);
//      final Node node = nodeCalcer.createNode();
//
//      for (int nodeIdx = nodeCalcer.getStateStart(); nodeIdx < nodeCalcer.getStateEnd(); nodeIdx++) {
//        final double dTds_i = dT.get(nodeIdx);
//        if (Math.abs(dTds_i) < MathTools.EPSILON || Math.abs(state.get(nodeIdx)) < MathTools.EPSILON)
//          continue;
//
//        final Vec curWGrad = nodeCalcer.getWeight(wGrad, nodeIdx);
//        final Vec stateGrad = nodeCalcer.getState(prevLayerGrad, nodeIdx);
//
//        final Vec curWeights = nodeCalcer.getWeight(allWeights, nodeIdx);
//        final Vec curState = nodeCalcer.getState(state, nodeIdx);
//
//        node.gradByStateTo(curState, curWeights, stateGrad);
//        node.gradByParametersTo(curState, curWeights, curWGrad);
//
//        scale(stateGrad, dTds_i);
//        scale(curWGrad, dTds_i);
//      }
//    }
//
//    VecTools.assign(to, wGrad);
//
//    return to;
  }

  long counter;
  protected void produceState(Seq<NodeCalcer> calcers, Vec weights, Vec state) {
    counter = 0;
    final int parallelism = ForkJoinPool.getCommonPoolParallelism();
    final int[] cursor = new int[parallelism + 1];
    final CountDownLatch latch = new CountDownLatch(parallelism);
    final int steps = calcers.length() / parallelism;
    for (int t = 0; t < parallelism; t++) {
      final int thread = t;
      ForkJoinPool.commonPool().execute(() -> {
        int i = 0;
        while(i < steps) {
          final int nodeIdx = thread + i * parallelism;
          if (nodeIdx > state.length())
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
            if (cursor[0] + 1 < end) {
              Thread.yield();
            }
          }
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

  public interface NodeCalcer {
    double apply(Vec state, Vec betta, int nodeIdx);
    int start(int nodeIdx);
    int end(int nodeIdx);
    void gradByStateTo(Vec state, Vec betta, Vec to);
    void gradByParametersTo(Vec state, Vec betta, Vec to);
  }
}
