package com.spbsu.ml.models.pgm;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
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
    this.bindings = bindings;
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
        TIntArrayList scratch = new TIntArrayList(route.length() + bindings.length);
        List<List<Route>> variants = new ArrayList<List<Route>>();
        int[] allRoutesIndex = new int[variants.size()];

        for (int i = 0; i < route.length(); i++) {
          if (isComposite(route.dst(i))) {
            scratch.add(-1);
            variants.add(bindings[i].routeInices[route.dst(i+1)]);
          }
          else scratch.add(route.dst(i));
        }
        List<Route> intermid = new ArrayList<Route>(variants.size());
        double prob = 0;
        while (increment(allRoutesIndex, variants)) {
          for (int i = 0; i < variants.size(); i++) {
            final Route interRoute = variants.get(i).get(allRoutesIndex[i]);
            intermid.set(i, interRoute);
            prob *= interRoute.p();
          }
          CompositeRoute croute = new CompositeRoute(scratch, intermid, prob);
          if (act.accept(croute))
            return true;
        }
        return false;
      }

      private boolean increment(int[] allRoutesIndex, List<List<Route>> variants) {
        for (int i = allRoutesIndex.length - 1; i >= 0; i--) {
          if (++allRoutesIndex[i] == variants.size()) {
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
    private final List<Route> intermediate;

    public CompositeRoute(TIntArrayList nodes, List<Route> intermediate, double prob) {
      this.intermediate = intermediate;
      this.prob = prob;
      int intIndex = 0;
      for (int i = 0; i < nodes.size(); i++) {
        final int current = nodes.get(i);
        if (current >= 0) {
          this.nodes.add(current);
        }
        else {
          final Route intRoute = intermediate.get(intIndex++);
          for (int j = 0; j < intRoute.length(); i++) {
            this.nodes.add(intIndex << 16 | j);
          }
        }
      }
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
      final int nodeIndex = nodes.get(stepNo) & 0xFFFF;
      final int intermediateIndex = nodeIndex >> 16;
      return intermediateIndex > 0 ? intermediate.get(intermediateIndex).dstOwner(nodeIndex) : CompositePGM.this;
    }

    @Override
    public int dst(int stepNo) {
      return nodes.get(stepNo) & 0xFFFF;
    }
  }

  @Override
  public Route next(FastRandom rng) {
    TIntArrayList nodes = new TIntArrayList();
    List<Route> intermediate = new ArrayList<Route>();
    int current = 0;
    double prob = 1;
    do {
      if (isComposite(current)){
        final Route inter = bindings[current].extension.next(rng);
        intermediate.add(inter);
        nodes.add(-1);
        prob *= inter.p();
        current = bindings[current].finalNodesBackBinding[inter.last()];
      }
      else {
        int next = rng.nextSimple(topology.row(current));
        prob *= topology.get(current, next);
        current = next;
      }
      nodes.add(current);
    }
    while(!isFinal(current));
    return new CompositeRoute(nodes, intermediate, prob);
  }

  public boolean isComposite(int node) {
    return bindings[node] != null;
  }
}
