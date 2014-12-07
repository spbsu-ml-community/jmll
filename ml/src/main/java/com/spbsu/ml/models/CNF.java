package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.text.NumberFormat;
import java.util.*;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class CNF extends RegionBase {
  private final Clause[] clauses;

  public CNF(Clause[] conditions, double inside, BFGrid grid) {
    super(grid, inside, 0.);
    this.clauses = conditions;
  }

  public boolean contains(Vec point) {
    for (Clause clause : clauses) {
      if (!clause.contains(point)) {
        return false;
      }
    }
    return true;
  }

  public boolean contains(BinarizedDataSet bds, int pindex) {
    for (Clause clause : clauses) {
      if (!clause.contains(bds, pindex)) {
        return false;
      }
    }
    return true;
  }

  public static class Clause extends RegionBase implements BinOptimizedModel {
    public final Condition[] conditions;

    public Clause(BFGrid grid, Condition... conditions) {
      super(grid, 1., 0);
      @SuppressWarnings("unchecked")
      final List<Condition> optimizedConditions = new ArrayList<>(conditions.length);
      for(int i = 0; i < conditions.length; i++) {
        final Condition next = conditions[i];
        boolean optimizedOut = false;
        for (int j = 0; j < optimizedConditions.size() && !optimizedOut; j++) {
          final Condition condition = optimizedConditions.get(j);
          if (condition.feature == next.feature) {
            condition.used.or(next.used);
            optimizedOut = true;
          }
        }
        if (!optimizedOut)
          optimizedConditions.add(new Condition(next.feature, (BitSet)next.used.clone()));
      }
      this.conditions = optimizedConditions.size() != conditions.length ? optimizedConditions.toArray(new Condition[optimizedConditions.size()]) : conditions;
    }

    @Override
    public boolean contains(BinarizedDataSet bds, int pindex) {
      if (conditions.length == 0)
        return true;
      for(int i = 0; i < conditions.length; i++) {
        final Condition condition = conditions[i];
        if (condition.contains(bds, pindex))
          return true;
      }
      return false;
    }

    @Override
    public boolean contains(Vec x) {
      if (conditions.length == 0)
        return true;
      for(int i = 0; i < conditions.length; i++) {
        final Condition condition = conditions[i];
        if (condition.contains(x))
          return true;
      }
      return false;
    }

    @Override
    public String toString() {
      return Arrays.toString(conditions);
    }

    public double cardinality() {
      double cardinality = conditions.length;
      for(int i = 0; i < conditions.length; i++) {
        cardinality += conditions[i].cardinality();
      }
      return cardinality;
    }
  }

  public static class Condition {
    public final int findex;
    public final BFGrid.BFRow feature;
    public final BitSet used;

    public Condition(BFGrid.BFRow feature, BitSet bins) {
      this.findex = feature.origFIndex;
      this.feature = feature;
      this.used = bins;
    }

    public int cardinality() {
      int prev = -1;
      int cardinality = 0;
      for (int i = 0; i < used.cardinality(); i++) {
        final int next = used.nextSetBit(prev + 1);
        if (next != prev+1) {
          cardinality += 2;
        }
        else if(prev < 0)
          cardinality++;
        prev = next;
      }
      if (prev == feature.size())
        cardinality--;
      return cardinality;
    }

    @Override
    public String toString() {
      final StringBuilder builder = new StringBuilder();
      builder.append("f[").append(feature.origFIndex).append("] \\in ");
      final NumberFormat pp = MathTools.numberFormatter();
      int prev = -1;
      for (int next = used.nextSetBit(0); next >= 0; next = used.nextSetBit(next+1)) {
        if (next != prev+1) {
          if (prev >= 0)
            builder.append(",").append(pp.format(feature.condition(prev))).append("]");
          builder.append("(").append(pp.format(feature.condition(next - 1)));
        }
        else if (prev < 0) {
          builder.append("(-\\infty");
        }
        prev = next;
      }
      if (prev == feature.size())
        builder.append(",+\\infty)");
      else
        builder.append(",").append(pp.format(feature.condition(prev))).append("]");
      return builder.toString();
    }

    public boolean contains(BinarizedDataSet bds, int pindex) {
      return used.get(bds.bins(findex)[pindex]);
    }

    public boolean contains(Vec x) {
      final double val = x.get(findex);
      return used.get(feature.bin(val));
    }
  }
}
