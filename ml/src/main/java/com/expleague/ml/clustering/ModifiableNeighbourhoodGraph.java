package com.expleague.ml.clustering;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

public class ModifiableNeighbourhoodGraph {
  private final List<Vec> points;
  private final List<TIntList> neighbours;
  private final int M;
  int k = 0;

  public ModifiableNeighbourhoodGraph(int m) {
    points = new ArrayList<>();
    neighbours = new ArrayList<>();
    M = m;
  }

  public Vec get(int i) {
    return points.get(i);
  }

  public int append(Vec v) {
    final int[] results = new int[M];
    final double[] distances = new double[M];
    final int knearest = knearest(v, 0, M, results, distances);
    points.add(v);
    neighbours.add(new TIntArrayList(results).subList(0, knearest));
    return points.size() - 1;
  }

  public boolean update(int idx, Vec vec) {
    final int[] results = new int[M + 1];
    final double[] distances = new double[M + 1];
    final int knearest = knearest(vec, idx, M + 1, results, distances);
    points.set(idx, vec);
    final TIntList newNn = new TIntArrayList(results).subList(1, knearest);
    return !newNn.equals(neighbours.set(idx, newNn));
  }

  public int nearest(Vec vec) {
    final int[] result = new int[1];
    knearest(vec, 0, 1, result, new double[1]);
    return result[0];
  }

  public int nearest(Vec vec, int[] result, double[] distances) {
    return knearest(vec, 0, result.length, result, distances);
  }

  private int knearest(Vec q, int start, int k, int[] result, double[] distances) {
    Arrays.fill(distances, Double.POSITIVE_INFINITY);
    Arrays.fill(result, -1);
    int count = 1;
    TIntArrayList candidates = new TIntArrayList(100);
    BitSet visited = new BitSet(this.k);
    result[0] = start;
    distances[0] = VecTools.distance(q, points.get(start));
    visited.set(start);
    candidates.add(start);
    while (candidates.size() > 0) {
      final int candidate = candidates.removeAt(candidates.size() - 1);
      final TIntList neighbourhood = neighbours.get(candidate);
      final int size = neighbourhood.size();
      boolean changed = false;
      for (int i = 0; i < size; i++) {
        final int next = neighbourhood.get(i);
        if (visited.get(next))
          continue;
        visited.set(next);
        final double nextDist = VecTools.distance(points.get(next), q);
        int idx = count - 1;
        while (idx >= 0 && nextDist < distances[idx]) { // bubble
          if (idx < k - 1) {
            distances[idx + 1] = distances[idx];
            result[idx + 1] = result[idx];
            if (idx == count - 1)
              count++;
          }
          distances[idx] = nextDist;
          changed = true;
          idx--;
          if (candidates.getQuick(candidates.size() - 1) != next)
            candidates.add(next);
        }
      }
      if (!changed)
        break;
    }
    return count;
  }

  public Vec[] points() {
    return points.toArray(new Vec[points.size()]);
  }
}
