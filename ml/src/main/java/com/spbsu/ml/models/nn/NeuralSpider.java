package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.TransC1;
import com.spbsu.ml.func.generic.WSum;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.IOException;
import java.io.Writer;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 25.05.15
 * Time: 12:57
 */
public abstract class NeuralSpider<T, S extends Seq<T>> {

  private final ThreadLocalArrayVec stateCache = new ThreadLocalArrayVec();
  public Vec compute(S argument, Vec weights) {
    final Topology topology = topology(argument, false);
    final Vec state = this.stateCache.get(topology.length());
    return produceState(topology, weights, state);
  }

  private final ThreadLocalArrayVec gradientSCache = new ThreadLocalArrayVec();

  public Vec parametersGradient(S argument, TransC1 target, Vec allWeights, Vec to) {
    final Topology topology = topology(argument, true);
    final Vec dT = gradientSCache.get(topology.length());
    final Vec state = stateCache.get(topology.length());
    final Vec output = produceState(topology, allWeights, state);

    target.gradientTo(output, dT.sub(dT.length() - topology.outputCount(), topology.outputCount()));
    final Vec prevLayerGrad = new SparseVec(state.dim());
    final Vec wGrad = new SparseVec(allWeights.dim());
    for (int nodeIndex = topology.length() - 1; nodeIndex > 0; nodeIndex--) {
      final double dTds_i = dT.get(nodeIndex);
      if (Math.abs(dTds_i) < MathTools.EPSILON || Math.abs(state.get(nodeIndex)) < MathTools.EPSILON)
        continue;

      final Node node = topology.at(nodeIndex);
      fill(prevLayerGrad, 0.);
      fill(wGrad, 0.);

      node.transByParameters(allWeights).gradientTo(state, prevLayerGrad);
      node.transByParents(state).gradientTo(allWeights, wGrad);

      scale(prevLayerGrad, dTds_i);
      scale(wGrad, dTds_i);

      append(dT, prevLayerGrad);
      append(to, wGrad);
    }
    return to;
  }

  protected Vec produceState(Topology topology, Vec weights) {
    return produceState(topology, weights, stateCache.get(topology.length()));
  }

  protected Vec produceState(Topology topology, Vec weights, Vec state) {
    state.set(0, 1.);
    final int nodesCount = topology.length();
    for (int nodeIndex = 1; nodeIndex < nodesCount; nodeIndex++) {
      if (topology.isDroppedOut(nodeIndex))
        continue;
      final Node node = topology.at(nodeIndex);
      final double value = node.transByParameters(weights).value(state);
      state.set(nodeIndex, value);
    }
    return state.sub(state.length() - topology.outputCount(), topology.outputCount());
  }

  protected abstract Topology topology(S seq, boolean dropout);

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

  public interface Node {
    FuncC1 transByParameters(Vec betta);
    FuncC1 transByParents(Vec state);
  }

  public interface Topology extends Seq<Node> {
    int outputCount();
    boolean isDroppedOut(int nodeIndex);

    public abstract class Stub extends Seq.Stub<Node> implements Topology {
      @Override
      public final boolean isImmutable() {
        return true;
      }

      @Override
      public final Seq<Node> sub(int start, int end) {
        throw new NotImplementedException();
      }

      @Override
      public final Class<Node> elementType() {
        return Node.class;
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
      return spider.topology(item, false);
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
