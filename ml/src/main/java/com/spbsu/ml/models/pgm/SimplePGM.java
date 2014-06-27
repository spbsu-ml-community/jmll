package com.spbsu.ml.models.pgm;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.impl.basis.IntBasis;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.Func;
import gnu.trove.list.array.TByteArrayList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

/**
 * User: solar
 * Date: 27.01.14
 * Time: 10:19
 */
public class SimplePGM extends Func.Stub implements ProbabilisticGraphicalModel {
  public static final double KNOWN_ROUTES_PROBABILITY = 0.999;
  public static final double MIN_SINGLE_ROUTE_PROBABILITY = 0.000001;
  public final Mx topology;
  private final boolean[] isFinalState;

  private final byte[] nodes;
  private final MyLWRoute[] routes;

  private double knownRoutesProBab;

  public SimplePGM(Mx topology) {
    topology = VecTools.copy(topology);
    isFinalState = new boolean[topology.rows()];
    for (int i = 0; i < topology.rows(); i++) {
      VecTools.normalizeL1(topology.row(i));
      if (VecTools.norm1(topology.row(i)) < MathTools.EPSILON)
        isFinalState[i] = true;
    }

    this.topology = topology;
    knownRoutesProBab = 0.;
    final List<MyRoute> order = new LinkedList<MyRoute>();
    final TreeSet<MyRoute> increment = new TreeSet<MyRoute>(new Comparator<MyRoute>() {
      @Override
      public int compare(MyRoute o1, MyRoute o2) {
        return o2.p() >= o1.p() ? 1 : -1;
      }
    });
    order.add(new MyRoute(new byte[]{0}, 1.));
    double minProbab = 0.01;
    final TByteArrayList nodes = new TByteArrayList();
    final TIntArrayList boundaries = new TIntArrayList();
    final TDoubleArrayList probabs = new TDoubleArrayList();

    while (knownRoutesProBab < KNOWN_ROUTES_PROBABILITY && minProbab > MIN_SINGLE_ROUTE_PROBABILITY) {
      boolean hasInc = false;
      final ListIterator<MyRoute> it = order.listIterator();
      increment.clear();
      while(it.hasNext()) {
        final MyRoute next = it.next();

        if (next.maxClosedProbab > minProbab) {
          next.disclose(topology, increment, minProbab);
          hasInc = true;
        }
        if (next.maxClosedProbab == 0.) {
          if (isFinal(next.last())) {
            probabs.add(next.probab);
            nodes.addAll(next.route);
            boundaries.add(nodes.size());
            knownRoutesProBab += next.probab;
          }
          it.remove();
        }
        if (!increment.isEmpty() && (!it.hasNext() || (order.get(it.nextIndex()).p() < increment.first().p()))) {
          it.add(increment.pollFirst());
          it.previous();
        }
      }

      if (!hasInc)
        minProbab /= 2.;
    }
    this.nodes = nodes.toArray();
    routes = new MyLWRoute[boundaries.size()];
    int start = 0;
    for (int i = 0; i < boundaries.size(); i++) {
      routes[i] = new MyLWRoute(start, start = boundaries.get(i), probabs.get(i));
    }
    System.out.println("Routes built: " + boundaries.size() + " weight: " + knownRoutesProBab);
  }

  /** need to change this to prefix tree which is way faster */
  public void visit(final Filter<Route> act, final int... controlPoints) {
    visit(new Filter<Route>() {
      @Override
      public boolean accept(Route route) {
        int index = 0;
        int controlPoint = controlPoints[index];
        for (int t = 0; t < route.length(); t++) {
          if (route.dst(t) == controlPoint) {
            index++;
            if (index == controlPoints.length)
              return act.accept(route);
            controlPoint = controlPoints[index];
          }
        }
        return false;
      }
    });
  }

  @Override
  public void visit(final Filter<Route> act) {
    for (MyLWRoute route : routes) {
      if (act.accept(route))
        return;
    }
  }

  @Override
  public double p(int... controlPoints) {
    final double[] result = new double[]{0.};
    visit(new Filter<Route>() {
      public boolean accept(Route route) {
        result[0] += route.p();
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
    return toInt.toArray();
  }

  public Vec next(FastRandom rng) {
    while(true) {
      SparseVec<IntBasis> result = new SparseVec<IntBasis>(new IntBasis(100));
      int next = 1;
      int index = 0;

      while (index < result.dim()) {
        next = rng.nextSimple(topology.row(next - 1)) + 1;
        result.set(index++, next);
        if (isFinalState[next -1])
          break;
      }
      if (result.get(result.dim()) == 0.)
        return result;
    }
  }

  @Override
  public int dim() {
    return topology.rows();
  }

  @Override
  public boolean isFinal(int node) {
    return isFinalState[node];
  }

  @Override
  public int knownRoutesCount() {
    return routes.length;
  }

  public Route knownRoute(int randRoute) {
    return routes[randRoute];
  }

  public double knownRouteWeight() {
    return knownRoutesProBab;
  }

  public class MyLWRoute implements Route {
    private int start, end;
    private double probab;

    private MyLWRoute(int start, int end, double probab) {
      this.start = start;
      this.end = end;
      this.probab = probab;
    }

    @Override
    public double p() {
      return probab;
    }

    @Override
    public int last() {
      return nodes[end - 1];
    }

    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder(20);
      builder.append('(');
      for (int i = 0; i < length(); i++) {
        if (i > 0)
          builder.append(',');
        builder.append(dst(i));
      }
      builder.append(')').append("->").append(p());
      return builder.toString();
    }

    @Override
    public int length() {
      return end - start;
    }

    @Override
    public ProbabilisticGraphicalModel dstOwner(int stepNo) {
      return SimplePGM.this;
    }

    @Override
    public int dst(int stepNo) {
      return nodes[stepNo + start];
    }
  }

  public class MyRoute implements Route {
    private final byte[] route;
    private final double probab;
    private double maxClosedProbab;

    public MyRoute(byte[] route, double probab) {
      this.route = route;
      this.probab = probab;

      final byte last = route[route.length - 1];
      maxClosedProbab = 0.;
      for (int next = 0; next < topology.columns(); next++) {
        maxClosedProbab = Math.max(probab * topology.get(last, next), maxClosedProbab);
      }
    }

    void disclose(Mx topology, TreeSet<MyRoute> container, double minProbab) {
      double lowerBound = maxClosedProbab;
      maxClosedProbab = 0.;

      for (int next = 0; next < topology.columns(); next++) {
        final double nextProbab = p() * topology.get(last(), next);
        if (nextProbab > lowerBound) // already disclosed
          continue;
        if (nextProbab < minProbab) {
          maxClosedProbab = Math.max(nextProbab, maxClosedProbab);
          continue;
        }

        final byte[] nextRouteB = new byte[route.length + 1];
        System.arraycopy(route, 0, nextRouteB, 0, route.length);
        nextRouteB[nextRouteB.length - 1] = (byte)next;
        container.add(new MyRoute(nextRouteB, nextProbab));
      }
    }

    @Override
    public double p() {
      return probab;
    }

    @Override
    public int last() {
      return route[route.length - 1];
    }

    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder(20);
      builder.append('(');
      for (int i = 0; i < length(); i++) {
        if (i > 0)
          builder.append(',');
        builder.append(dst(i));
      }
      builder.append(')').append("->").append(p());
      return builder.toString();
    }

    @Override
    public int length() {
      return route.length;
    }

    @Override
    public ProbabilisticGraphicalModel dstOwner(int stepNo) {
      return SimplePGM.this;
    }

    @Override
    public int dst(int stepNo) {
      return route[stepNo];
    }
  }
}
