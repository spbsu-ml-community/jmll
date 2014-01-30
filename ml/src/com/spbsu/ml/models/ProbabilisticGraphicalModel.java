package com.spbsu.ml.models;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.IntBasis;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.Func;
import gnu.trove.TIntArrayList;

import java.util.*;

/**
 * User: solar
 * Date: 27.01.14
 * Time: 10:19
 */
public class ProbabilisticGraphicalModel extends Func.Stub {
  public static final double KNOWN_ROUTES_PROBABILITY = 0.999;
  public static final double MIN_SINGLE_ROUTE_PROBABILITY = 0.00001;
  public final Mx topology;
  private final Route[] knownRoutes;
  private double knownRoutesProBab;

  public ProbabilisticGraphicalModel(Mx topology) {
    topology = VecTools.copy(topology);
    for (int i = 0; i < topology.rows(); i++) {
      VecTools.normalizeL1(topology.row(i));
    }

    this.topology = topology;
    knownRoutesProBab = 0.;
    final List<Route> order = new LinkedList<Route>();
    final TreeSet increment = new TreeSet<Route>(new Comparator<Route>() {
      @Override
      public int compare(Route o1, Route o2) {
        return o2.probab >= o1.probab ? 1 : -1;
      }
    });
    order.add(new Route(new byte[]{0}, 1.));
    double minProbab = 0.01;
    while (knownRoutesProBab < KNOWN_ROUTES_PROBABILITY && minProbab > MIN_SINGLE_ROUTE_PROBABILITY) {
      boolean hasInc = false;
      final ListIterator<Route> it = order.listIterator();
      Route next = it.next();
      Route orderNext = it.hasNext() ? it.next() : null;
      increment.clear();
      do {
        knownRoutesProBab += next.disclose(topology, increment, minProbab);
        next = null;
        if (orderNext != null && (increment.size() == 0 || orderNext.probab >= ((Route)increment.first()).probab)) {
          next = orderNext;
          orderNext = it.hasNext() ? it.next() : null;
        }
        else if (increment.size() > 0) {
          hasInc = true;
          next = (Route)increment.pollFirst();
          it.add(next);
        }
      } while (next != null);

      if (!hasInc)
        minProbab /= 2.;
    }
    { // cleaning up
      Iterator<Route> it = order.iterator();
      while(it.hasNext()) {
        if (it.next().last() != topology.rows() - 1)
          it.remove();
      }
    }
    knownRoutes = order.toArray(new Route[order.size()]);
  }

  /** need to change this to prefix tree which is way faster */
  public void visit(Filter<Route> act, int... controlPoints) {
    for (int i = 0; i < knownRoutes.length; i++) {
      final Route route = knownRoutes[i];
      int index = 0;
      for (int t = 0; t < route.nodes.length && index < controlPoints.length; t++) {
        if (route.nodes[t] == controlPoints[index])
          index++;
      }
      if (index == controlPoints.length && act.accept(route))
        return;
    }
  }

  public double p(int... controlPoints) {
    final double[] result = new double[]{0.};
    visit(new Filter<Route>() {
      public boolean accept(Route route) {
        result[0] += route.probab;
        return false;
      }
    }, controlPoints);
    return result[0];
  }

  @Override
  public double value(Vec x) {
    return p(extractControlPoints(x));
  }

  public int[] extractControlPoints(Vec x) {
    TIntArrayList toInt = new TIntArrayList(x.dim() + 1);
    toInt.add(0);
    for (int i = 0; i < x.dim(); i++) {
      final double next = x.get(i);
      if (next < 1 || next >= topology.rows())
        break;
      toInt.add((int)next - 1);
    }
    return toInt.toNativeArray();
  }

  public Vec next(FastRandom rng) {
    SparseVec<IntBasis> result = new SparseVec<IntBasis>(new IntBasis(100));
    int next = 1;
    int index = 0;
    do {
      next = rng.nextSimple(topology.row(next - 1)) + 1;
      result.set(index++, next);
    }
    while (next < topology.rows());
    return result;
  }

  @Override
  public int dim() {
    return topology.rows();
  }

  public static class Route {
    public byte[] nodes;
    public double probab;
    private double disclosedP = 1.;

    public Route(byte[] route, double proBab) {
      nodes = route;
      probab = proBab;
    }

    private double disclose(Mx topology, TreeSet<Route> container, double minProbab) {
      if (minProbab >= disclosedP)
        return 0.;
      double disclosedProbab = 0;
      for (int next = 0; next < topology.columns(); next++) {
        final double v = topology.get(nodes[nodes.length - 1], next);
        if (probab * v < minProbab || probab * v > disclosedP)
          continue;

        final byte[] route = new byte[nodes.length + 1];
        System.arraycopy(nodes, 0, route, 0, nodes.length);
        route[route.length - 1] = (byte)next;
        final Route nextRoute = new Route(route, probab * v);
        container.add(nextRoute);
        if (next == topology.columns() - 1) {
          disclosedProbab += probab * v;
          nextRoute.disclosedP = 0.;
        }
      }
      disclosedP = minProbab;
      return disclosedProbab;
    }

    public int last() {
      return nodes[nodes.length - 1];
    }

    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder(20);
      builder.append('(');
      for (int i = 0; i < nodes.length; i++) {
        if (i > 0)
          builder.append(',');
        builder.append((int)nodes[i]);
      }
      builder.append(')').append("->").append(probab);
      return builder.toString();
    }
  }
}
