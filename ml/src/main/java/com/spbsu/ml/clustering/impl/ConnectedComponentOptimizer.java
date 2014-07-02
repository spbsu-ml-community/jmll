package com.spbsu.ml.clustering.impl;

import com.spbsu.commons.func.Computable;
import com.spbsu.ml.clustering.ClusterizationAlgorithm;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.util.RBTreeNode;
import com.spbsu.commons.util.RBTreeNodeBase;
import com.spbsu.commons.util.RedBlackTree;


import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.procedure.TObjectProcedure;
import org.jetbrains.annotations.NotNull;

import java.util.*;

/**
 * User: solar
 * Date: 14.02.2010
 * Time: 0:48:33
 */
public class ConnectedComponentOptimizer<T> implements ClusterizationAlgorithm<T> {
  private ClusterizationAlgorithm<T> algorithm;
  private double minToJoin;

  public ConnectedComponentOptimizer(ClusterizationAlgorithm<T> algorithm, double minToJoin) {
    this.algorithm = algorithm;
    this.minToJoin = minToJoin;
  }

  private static class IndexedVecIter <T> {
    VecIterator iter;
    T t;
    int componentIndex;

    private IndexedVecIter(VecIterator iter, T t, int index) {
      this.iter = iter;
      this.t = t;
      componentIndex = index;
    }
  }

  private static class VecIterEntry extends RBTreeNodeBase {
    List<IndexedVecIter> iters = new LinkedList<IndexedVecIter>();
    final int index;

    public VecIterEntry(int index) {
      this.index = index;
    }

    @Override
    public int compareTo(RBTreeNode node) {
      final VecIterEntry entry = (VecIterEntry) node;
      return index - entry.index;
    }
  }

  private static void processIter(Set<VecIterEntry> iters, TIntObjectHashMap<VecIterEntry> cache, IndexedVecIter iter) {
    final int index = iter.iter.index();
    VecIterEntry iterEntry = cache.get(index);
    if (iterEntry == null) {
      iterEntry = new VecIterEntry(index);
      iters.add(iterEntry);
      cache.put(index, iterEntry);
    }
    iterEntry.iters.add(iter);
  }

  @NotNull
  @Override
  public Collection<? extends Collection<T>> cluster(Collection<T> dataSet, final Computable<T, Vec> data2DVector) {
    final RedBlackTree<VecIterEntry> iters = new RedBlackTree<VecIterEntry>();
    final TIntObjectHashMap<VecIterEntry> cache = new TIntObjectHashMap<VecIterEntry>();
    final List<IndexedVecIter<T>> entries = new ArrayList<IndexedVecIter<T>>();
    double minToJoin = this.minToJoin;// + 0.5 * (1 - Math.min(1,  Math.log(2000) / Math.log(dataSet.size())));
    {
      int index = 1;
      for (T t : dataSet) {
        final Vec vec = data2DVector.compute(t);
        final VecIterator iter = vec.nonZeroes();
        while (iter.advance() && iter.value() < minToJoin);
        if (iter.isValid()) {
          final IndexedVecIter<T> entry = new IndexedVecIter<T>(iter, t, index++);
          entries.add(entry);
          processIter(iters, cache, entry);
        }
      }
    }
    while (!iters.isEmpty()) {
      final VecIterEntry topEntry = iters.pollFirst();
      int maxComponentIndex = 0;
      final boolean join = topEntry.iters.size() > 1 && topEntry.iters.size() < dataSet.size() / 10;

      if (join) {
        double sum = 0;
        int count = 0;
        int prev = 0;
        for (IndexedVecIter iter : topEntry.iters) {
          count++;
          sum += iter.iter.value();
          if (prev != 0 && prev != iter.iter.index())
            System.err.println("FUCK!!!");
          prev = iter.iter.index();
          if (iter.componentIndex > maxComponentIndex) {
            maxComponentIndex = iter.componentIndex;
          }
        }
//        System.out.println(termsBasis.fromIndex(topEntry.iters.at(0).iter.index()) + ": " + topEntry.iters.size()+ ":" + maxComponentIndex + ":" + (sum / Math.max(1, count)));
      }

      for (IndexedVecIter iter : topEntry.iters) {
        if (join)
          iter.componentIndex = maxComponentIndex;
        while (iter.iter.advance() && iter.iter.value() < minToJoin);
        if (iter.iter.isValid()) 
          processIter(iters, cache, iter);
      }

    }
    TIntObjectHashMap<Set<T>> components = new TIntObjectHashMap<Set<T>>();
    for (IndexedVecIter<T> entry : entries) {
      Set<T> component = components.get(entry.componentIndex);
      if (component == null)
        components.put(entry.componentIndex, component = new HashSet<T>());
      component.add(entry.t);
    }

//    System.out.println(components.size() + " components found");

    final List<Collection<T>> clusters = new ArrayList<Collection<T>>();
    components.forEachValue(new TObjectProcedure<Set<T>>() {
      @Override
      public boolean execute(Set<T> ts) {
        for (Collection<T> cluster : algorithm.cluster(ts, data2DVector)) {
          clusters.add(cluster);
        }
        return true;
      }
    });
    return clusters;
  }
}
