package com.expleague.ml.models.nn;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.TransC1;
import com.expleague.ml.func.generic.WSum;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.io.Writer;
import java.util.concurrent.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.expleague.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 12:57
 */
public abstract class NeuralSpider<T, S extends Seq<T>> {

  private final ThreadLocalArrayVec stateCache = new ThreadLocalArrayVec();

  public Vec compute(S argument, Vec weights) {
    final Topology topology = topology(false);
    final Vec state = this.stateCache.get(topology.dim());
    final Vec arg = new ArrayVec(argument.toArray());
    for (int i = 0; i < arg.length(); i++) {
      state.set(i, arg.get(i));
    }
    return produceState(topology, weights, state);
  }

  private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

  public Vec parametersGradient(S argument, TransC1 target, Vec allWeights, Vec to) {
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
//    final Vec prevLayerGrad = new SparseVec(state.dim());
//    final Vec wGrad = new SparseVec(allWeights.dim());
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

  protected Vec produceState(Topology topology, Vec weights) {
    return produceState(topology, weights, stateCache.get(topology.length()));
  }

  protected Vec produceState(Topology topology, Vec weights, Vec state) {
    final int typesCount = topology.length();
    for (int topologIdx = 0; topologIdx < typesCount; topologIdx++) {
      final NodeCalcer nodeCalcer = topology.at(topologIdx);

      for (int nodeIdx = nodeCalcer.getStartNodeIdx(); nodeIdx < nodeCalcer.getEndNodeIdx(); nodeIdx++) {
        final double value = nodeCalcer.apply(state, weights, nodeIdx);
        state.set(nodeIdx, value);
      }
    }

    return state.sub(state.length() - topology.outputCount(), topology.outputCount());
  }

  public abstract int numParameters();

  protected abstract Topology topology(boolean dropout);

  public abstract int dim();

  @SuppressWarnings("UnusedDeclaration")
  public void printTopology(Topology topology, Vec allWeights, Writer to) throws IOException {
//    for (int i = 0; i < topology.nodesCount(); i++) {
//      to.append(Integer.toString(i)).append(" ");
//
//      for (int j = i + 1; j < topology.nodesCount(); j++) {
//        final IntSeq connections = topology.connections(j);
//        final Vec weights = topology.parameters(j, allWeights);
//        final int index = ArrayTools.indexOf(i, connections);
//        if(index >= 0)
//          to.append(" ").append(Integer.toString(j)).append(':').append(CharSeqTools.prettyPrint.format(weights.get(index)));
//      }
//      to.append("\n");
//    }
//    to.flush();
  }

  public Vec parametersGradient(S x, TransC1 target, Vec weights) {
    return parametersGradient(x, target, weights, new ArrayVec(weights.dim()));
  }

  public interface NodeCalcer {
    double apply(Vec state, Vec betta, int nodeIdx);
    int getStartNodeIdx();
    int getEndNodeIdx();
    void gradByStateTo(Vec state, Vec betta, Vec to);
    void gradByParametersTo(Vec state, Vec betta, Vec to);
  }

  public interface Topology extends Seq<NodeCalcer> {
    int outputCount();
    boolean isDroppedOut(int nodeIndex);
    int dim();

    abstract class Stub extends Seq.Stub<NodeCalcer> implements Topology {
      @Override
      public final boolean isImmutable() {
        return true;
      }

      @Override
      public final Seq<NodeCalcer> sub(int start, int end) {
        throw new NotImplementedException();
      }

      @Override
      public final Class<NodeCalcer> elementType() {
        return NodeCalcer.class;
      }

      @Override
      public Stream<NodeCalcer> stream() {
        return IntStream.range(0, length()).mapToObj(this::at);
      }
    }
  }

  public NeuralNet decisionByInput(S input) {
    return new NeuralNet<>(this, input);
  }

  public static class NeuralNet<T, S extends Seq<T>> extends TransC1.Stub {
    private final NeuralSpider<T, S> spider;
    private final S item;

    public NeuralNet(NeuralSpider<T, S> network, S row) {
      this.spider = network;
      this.item = row;
    }

    @Override
    public Vec transTo(Vec x, Vec to) {
      final Topology topology = topology();
      final Vec state = spider.produceState(topology, x);
      assign(to, state);
      return to;
    }

    public Vec gradientTo(Vec x, Vec to, FuncC1 target) {
      return spider.parametersGradient(item, target, x, to);
    }

    public Topology topology() {
      return spider.topology(false);
    }

    public Vec state(Vec x) {
      final Topology topology = topology();
      final ArrayVec state = new ArrayVec(topology.length());
      spider.produceState(topology, x, state);
      return state;
    }

    @Override
    public int xdim() {
      return spider.dim();
    }

    @Override
    public int ydim() {
      return topology().outputCount();
    }

    @Override
    public Vec gradientRowTo(Vec x, Vec to, int index) {
      final Vec w = new ArrayVec(ydim());
      w.set(index, 1);
      return spider.parametersGradient(item, new WSum(w), x, to);
    }
  }
}
