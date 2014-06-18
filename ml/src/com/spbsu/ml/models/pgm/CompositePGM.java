package com.spbsu.ml.models.pgm;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import gnu.trove.list.array.TByteArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.List;

/**
* User: solar
* Date: 27.01.14
* Time: 10:19
*/
public class CompositePGM extends SimplePGM {
  public static class PGMExtensionBinding {
    public final int node2bind2;
    public final int[] finalNodesBackBinding;
    public final ProbabilisticGraphicalModel extension;
    public final List<Route>[] routeInices;


    public PGMExtensionBinding(int node2bind2, ProbabilisticGraphicalModel extension, int[] finalNodesBackBinding) {
      this.node2bind2 = node2bind2;
      this.extension = extension;
      this.finalNodesBackBinding = new int[extension.dim()];
      //noinspection unchecked
      routeInices = new List[extension.dim()];
      int finalNodesIndex = 0;
      for (int i = 0; i < extension.dim(); i++) {
        if (extension.isFinal(i)) {
          routeInices[i] = new ArrayList<Route>();
          this.finalNodesBackBinding[i] = finalNodesBackBinding[finalNodesIndex++];
        }
        else
          this.finalNodesBackBinding[i] = -1;
      }
      extension.visit(new Filter<Route>() {
        @Override
        public boolean accept(Route route) {
          routeInices[route.last()].add(route);
          return false;
        }
      });
    }
  }

  private final PGMExtensionBinding[] bindings;

  public CompositePGM(Mx topology, PGMExtensionBinding... bindings) {
    super(bindExtensions(topology, bindings));
    this.bindings = new PGMExtensionBinding[dim()];
    for (int i = 0; i < bindings.length; i++) {
      this.bindings[bindings[i].node2bind2] = bindings[i];
    }
  }

  private static Mx bindExtensions(Mx topology, PGMExtensionBinding[] bindings) {
    for (int i = 0; i < bindings.length; i++) {
      PGMExtensionBinding binding = bindings[i];
      final Vec row = topology.row(binding.node2bind2);
      VecTools.scale(row, 0.);
      final Vec finals = new ArrayVec(binding.extension.dim());
      binding.extension.visit(new Filter<Route>() {
        @Override
        public boolean accept(Route route) {
          finals.adjust(route.last(), route.p());
          return false;
        }
      });
      for (int j = 0; j < binding.finalNodesBackBinding.length; j++) {
        row.set(binding.finalNodesBackBinding[j], finals.get(j));
      }
      VecTools.normalizeL1(row);
    }
    return topology;
  }

  @Override
  public void visit(final Filter<Route> act) {
    super.visit(new Filter<Route>() {
      @Override
      public boolean accept(Route route) {
        List<List<Route>> variants = new ArrayList<List<Route>>();

        for (int i = 0; i < route.length(); i++) {
          if (isComposite(route.dst(i))) {
            variants.add(bindings[route.dst(i)].routeInices[route.dst(i+1)]);
          }
        }
        int[] allRoutesIndex = new int[variants.size()];
        Route[] intermid = new Route[variants.size()];
        while (increment(allRoutesIndex, variants)) {
          for (int i = 0; i < variants.size(); i++) {
            final Route interRoute = variants.get(i).get(allRoutesIndex[i]);
            intermid[i] = interRoute;
          }
          CompositeRoute croute = new CompositeRoute(route, intermid);
          if (act.accept(croute))
            return true;
        }
        return false;
      }

      private boolean increment(int[] allRoutesIndex, List<List<Route>> variants) {
        for (int i = allRoutesIndex.length - 1; i >= 0; i--) {
          if (++allRoutesIndex[i] == variants.get(i).size()) {
            allRoutesIndex[i] = 0;
          }
          else return true;
        }
        return false;
      }
    });
  }

  public class CompositeRoute extends Route {
    private final double prob;
    private final TIntArrayList nodes = new TIntArrayList();
    private final Route[] intermediate;

    public CompositeRoute(Route original, Route[] intermediate) {
      this.intermediate = intermediate;
      int intIndex = 0;
      double prob = original.p();
      for (int i = 0; i < original.length(); i++) {
        final int current = original.dst(i);
        nodes.add(current);
        if (isComposite(current)) {
          final Route intRoute = intermediate[intIndex++];
          prob *= intRoute.p();
          for (int j = 0; j < intRoute.length(); j++) {
            nodes.add(intIndex << 16 | j);
          }
        }
      }
      this.prob = prob;
    }

    @Override
    public double p() {
      return prob;
    }

    @Override
    public int last() {
      return nodes.get(nodes.size() - 1);
    }

    @Override
    public int length() {
      return nodes.size();
    }

    @Override
    public ProbabilisticGraphicalModel dstOwner(int stepNo) {
      int nodeIndex = nodes.get(stepNo);
      final int intermediateIndex = (nodeIndex >> 16);
      nodeIndex &= 0xFFFF;
      return intermediateIndex > 0 ? intermediate[intermediateIndex - 1].dstOwner(nodeIndex) : CompositePGM.this;
    }

    @Override
    public int dst(int stepNo) {
      return nodes.get(stepNo) & 0xFFFF;
    }

    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder(20);
      builder.append('(');
      int intindex = 0;
      for (int i = 0; i < length(); i++) {
        if (i > 0)
          builder.append(',');
        if (dstOwner(i) != CompositePGM.this) {
          builder.append(intermediate[intindex++]);
          while(dstOwner(i+1) != CompositePGM.this) //skip all intermidiate nodes
            i++;
        }
        else builder.append(dst(i));
      }
      builder.append(')').append("->").append(p());
      return builder.toString();
    }
  }

  @Override
  public Route next(FastRandom rng) {
    TByteArrayList nodes = new TByteArrayList();
    List<Route> intermediate = new ArrayList<Route>();
    int current = 0;
    double prob = 1;
    do {
      if (isComposite(current)){
        final Route inter = bindings[current].extension.next(rng);
        intermediate.add(inter);
        current = bindings[current].finalNodesBackBinding[inter.last()];
      }
      else {
        int next = rng.nextSimple(topology.row(current));
        prob *= topology.get(current, next);
        current = next;
      }
      nodes.add((byte)current);
    }
    while(!isFinal(current));
    return new CompositeRoute(new MyRoute(nodes.toArray(), prob), intermediate.toArray(new Route[intermediate.size()]));
  }

  public boolean isComposite(int node) {
    return bindings[node] != null;
  }
}
