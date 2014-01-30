package com.spbsu.ml.models;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
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
    this.topology = topology;
    knownRoutesProBab = 0.;
    final TreeSet<Route> orderedSet = new TreeSet<Route>(new Comparator<Route>() {
      @Override
      public int compare(Route o1, Route o2) {
        return Double.compare(o1.probab, o2.probab);
      }
    });
    orderedSet.add(new Route(new byte[]{0}, 1.));
    double minProbab = 0.01;
    while (knownRoutesProBab < KNOWN_ROUTES_PROBABILITY && minProbab > MIN_SINGLE_ROUTE_PROBABILITY) {
      double increment = 0.;
      for (Route route : orderedSet) {
        if (route.disclosed)
          continue;
        if (minProbab > route.probab)
          break;
        double discProb = route.disclose(topology, orderedSet, minProbab);
        increment += discProb;
        if (discProb > 0)
          break;
      }
      if (increment == 0.)
        minProbab /= 2.;
    }
    Iterator<Route> it = orderedSet.iterator();
    while(it.hasNext()) {
      if (it.next().last() != topology.rows())
        it.remove();
    }
    knownRoutes = orderedSet.toArray(new Route[orderedSet.size()]);
  }

  /** need to change this to prefix tree which is way faster */
  public void visit(Action<Route> act, int... controlPoints) {
    for (int i = 0; i < knownRoutes.length; i++) {
      final Route route = knownRoutes[i];
      int index = 0;
      for (int t = 0; t < route.nodes.length && index < controlPoints.length; t++) {
        if (route.nodes[t] == controlPoints[index])
          index++;
      }
      if (index == controlPoints.length)
        act.invoke(route);
    }
  }

  public double p(int... controlPoints) {
    final double[] result = new double[]{0.};
    visit(new Action<Route>() {
      @Override
      public void invoke(Route route) {
        result[0] += route.probab;
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
      if (next < 1 || next >= topology.rows() - 1)
        break;
      toInt.add((int)next);
    }
    return toInt.toNativeArray();
  }

  @Override
  public int dim() {
    return topology.rows();
  }

  public static class Route {
    public byte[] nodes;
    public double probab;
    private boolean disclosed;

    public Route(byte[] route, double proBab) {
      nodes = route;
      probab = proBab;
    }

    private double disclose(Mx topology, Set<Route> container, double minProbab) {
      disclosed = true;
      double disclosedProbab = 0;
      for (int next = 0; next < topology.columns(); next++) {
        final double v = topology.get(nodes[nodes.length - 1], next);
        if (probab * v < minProbab)
          continue;

        final byte[] route = new byte[nodes.length + 1];
        System.arraycopy(nodes, 0, route, 0, nodes.length);
        container.add(new Route(route, probab * v));
        disclosedProbab += probab * v;
      }
      return disclosedProbab;
    }

    public int last() {
      return nodes[nodes.length - 1];
    }
  }
}
