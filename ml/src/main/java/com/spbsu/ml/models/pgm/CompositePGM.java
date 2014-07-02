//package com.spbsu.ml.models.pgm;
//
//import com.spbsu.commons.filters.Filter;
//import com.spbsu.commons.math.vectors.impl.basis.IntBasis;
//import com.spbsu.commons.math.vectors.Mx;
//import com.spbsu.commons.math.vectors.Vec;
//import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
//import com.spbsu.commons.random.FastRandom;
//import gnu.trove.list.array.TIntArrayList;
//
///**
// * User: solar
// * Date: 27.01.14
// * Time: 10:19
// */
//public class CompositePGM extends SimplePGM {
//  public static class PGMExtensionBinding {
//    public final int node2bind2;
//    public final int[] finalNodesBackBinding;
//    public final SimplePGM extension;
//
//    public PGMExtensionBinding(int node2bind2, SimplePGM extension, int[] finalNodesBackBinding) {
//      this.node2bind2 = node2bind2;
//      this.extension = extension;
//      this.finalNodesBackBinding = finalNodesBackBinding;
//      for (int i = 0; i < finalNodesBackBinding.length; i++) {
//        if(!extension.isFinal(finalNodesBackBinding[i]))
//          throw new IllegalArgumentException("finalNodesBackBinding contains non final node: " + finalNodesBackBinding[i]);
//      }
//    }
//  }
//
//  private final PGMExtensionBinding[] bindings;
//
//  public CompositePGM(Mx topology, PGMExtensionBinding... bindings) {
//    super(topology);
//    this.bindings = bindings;
//  }
//
//  /** need to change this to prefix tree which is way faster */
//  public void visit(Filter<Route> act, int... controlPoints) {
//    final int length = knownRoutes.length;
//    for (int i = 0; i < length; i++) {
//      final Route knownRoute = knownRoutes[i];
//      int index = 0;
//      for (int t = 0; t < knownRoute.nodes.length && index < controlPoints.length; t++) {
//        if (knownRoute.nodes[t] == controlPoints[index])
//          index++;
//      }
//      if (index == controlPoints.length && act.accept(knownRoute))
//        return;
//    }
//  }
//
//  public double p(int... controlPoints) {
//    final double[] result = new double[]{0.};
//    visit(new Filter<Route>() {
//      public boolean accept(Route knownRoute) {
//        result[0] += knownRoute.probab;
//        return false;
//      }
//    }, controlPoints);
//    return result[0];
//  }
//
//  @Override
//  public double value(Vec x) {
//    return p(extractControlPoints(x));
//  }
//
//  public int[] extractControlPoints(Vec x) {
//    TIntArrayList toInt = new TIntArrayList(x.dim() + 1);
//    toInt.add(0);
//    for (int i = 0; i < x.dim(); i++) {
//      final double next = x.at(i);
//      if (next < 1 || next >= topology.rows())
//        break;
//      toInt.add((int)next - 1);
//    }
//    return toInt.toArray();
//  }
//
//  public Vec next(FastRandom rng) {
//    while(true) {
//      SparseVec<IntBasis> result = new SparseVec<IntBasis>(new IntBasis(100));
//      int next = 1;
//      int index = 0;
//
//      while (index < result.dim()) {
//        next = rng.nextSimple(topology.row(next - 1)) + 1;
//        result.set(index++, next);
//        if (next == topology.columns())
//          break;
//      }
//      if (result.at(result.dim()) == 0.)
//        return result;
//    }
//  }
//}
