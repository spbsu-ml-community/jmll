package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.generic.Identity;
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
  private final ThreadLocalArrayVec gradientWCache = new ThreadLocalArrayVec();

  public Vec parametersGradient(S argument, FuncC1 target, Vec allWeights) {
    final Topology topology = topology(argument, true);
    final Vec dT = gradientSCache.get(topology.length());
    final Vec gradientW = gradientWCache.get(allWeights.dim());
    final Vec state = stateCache.get(topology.length());
    final Vec output = produceState(topology, allWeights, state);

    final Vec gradient0 = target.gradient(output);
    assign(dT.sub(dT.length() - gradient0.dim(), gradient0.dim()), gradient0);
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
      append(gradientW, wGrad);
    }
    return gradientW;
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
    if (state.get(state.dim() - 1) == 0)
      System.out.println();
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

  public static class NeuralNet<T, S extends Seq<T>> extends Trans.Stub implements FuncC1 {
    private final NeuralSpider<T, S> spider;
    private final S item;

    public NeuralNet(NeuralSpider<T, S> network, S row) {
      this.spider = network;
      this.item = row;
    }

    @Override
    public Vec gradientTo(Vec x, Vec to) {
      Mx result = new VecBasedMx(ydim(), xdim());
      for (int i = 0; i < ydim(); i++) {
        Vec weights = new ArrayVec(ydim());
        fill(weights, 0.);
        weights.set(i, 1.);
        Vec vec = spider.parametersGradient(item, new WSum(weights), x);
        for (int j = 0; j < xdim(); j++) {
          result.set(i, j, vec.get(j));
        }
      }
      assign(to, result);
      return to;
    }

    @Override
    public Vec gradient(Vec x) {
      final Vec result = new ArrayVec(xdim() * ydim());
      return gradientTo(x, result);
    }

    @Override
    public Trans gradient() {
      return new Trans.Stub() {
        @Override
        public int xdim() {
          return NeuralNet.this.xdim();
        }

        @Override
        public int ydim() {
          return NeuralNet.this.xdim() * NeuralNet.this.ydim();
        }

        @Override
        public Vec trans(Vec arg) {
          return NeuralNet.this.gradient(arg);
        }
      };
    }

    @Override
    public int xdim() {
      return dim();
    }

    @Override
    public int ydim() {
      return topology().outputCount();
    }

    @Override
    public Vec trans(Vec arg) {
      return spider.compute(item, arg);
    }

    @Override
    public double value(Vec x) {
      return spider.compute(item, x).get(0);
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
    public int dim() {
      return spider.dim();
    }
  }
}
