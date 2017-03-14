package com.spbsu.ml.models.pgm;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.Func;
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
  protected final boolean[] isFinalState;

  private final byte[] nodes;
  private final MyLWRoute[] routes;
  public double meanERouteLength = 0;

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
      public int compare(final MyRoute o1, final MyRoute o2) {
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

  public SimplePGM(final Mx next, final double meanLen) {
    this(next);
    this.meanERouteLength = meanLen;
  }

  /** need to change this to prefix tree which is way faster */
  public void visit(final Filter<Route> act, final int... controlPoints) {
    visit(new Filter<Route>() {
      @Override
      public boolean accept(final Route route) {
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
    for (final MyLWRoute route : routes) {
      if (act.accept(route))
        return;
    }
  }

  @Override
  public double p(final int... controlPoints) {
    final double[] result = new double[]{0.};
    visit(new Filter<Route>() {
      @Override
      public boolean accept(final Route route) {
        result[0] += route.p();
        return false;
      }
    }, controlPoints);
    return result[0];
  }

  @Override
  public double value(final Vec x) {
    return p(extractControlPoints(x));
  }

  public int[] extractControlPoints(final Vec x) {
    final TIntArrayList toInt = new TIntArrayList(x.dim() + 1);
    toInt.add(0);
    for (int i = 0; i < x.dim(); i++) {
      final double next = x.get(i);
      if (next < 1 || next >= topology.rows())
        break;
      toInt.add((int)next - 1);
    }
    return toInt.toArray();
  }

  @Override
  public Route next(final FastRandom rng) {
    final TByteArrayList result = new TByteArrayList(100);
    byte next = 0;
    double p = 1;

    while (true) {
      final Vec trans = topology.row(next);
      next = (byte)rng.nextSimple(trans);
      p *= trans.get(next);
      result.add(next);
      if (isFinalState[next])
        break;
    }
    return new MyRoute(result.toArray(), p);
  }

  @Override
  public int dim() {
    return topology.rows();
  }

  @Override
  public boolean isFinal(final int node) {
    return isFinalState[node];
  }

  @Override
  public int knownRoutesCount() {
    return routes.length;
  }

  @Override
  public Route knownRoute(final int randRoute) {
    return routes[randRoute];
  }

  @Override
  public double knownRoutesWeight() {
    return knownRoutesProBab;
  }

  public class MyLWRoute implements Route {
    private final int start;
    private final int end;
    private final double probab;

    private MyLWRoute(final int start, final int end, final double probab) {
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
    public int length() {
      return end - start;
    }

    @Override
    public ProbabilisticGraphicalModel dstOwner(final int stepNo) {
      return SimplePGM.this;
    }

    @Override
    public int dst(final int stepNo) {
      return nodes[stepNo + start];
    }
  }

  public class MyRoute implements Route {
    private final byte[] route;
    private final double probab;
    private double maxClosedProbab;

    public MyRoute(final byte[] route, final double probab) {
      this.route = route;
      this.probab = probab;

      final byte last = route[route.length - 1];
      maxClosedProbab = 0.;
      for (int next = 0; next < topology.columns(); next++) {
        maxClosedProbab = Math.max(probab * topology.get(last, next), maxClosedProbab);
      }
    }

    void disclose(final Mx topology, final TreeSet<MyRoute> container, final double minProbab) {
      final double lowerBound = maxClosedProbab;
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
    public int length() {
      return route.length;
    }

    @Override
    public ProbabilisticGraphicalModel dstOwner(final int stepNo) {
      return SimplePGM.this;
    }

    @Override
    public int dst(final int stepNo) {
      return route[stepNo];
    }
  }
}
