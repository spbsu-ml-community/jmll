package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.func.Ensemble;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.procedure.TObjectDoubleProcedure;
import gnu.trove.set.TIntSet;

import java.util.*;

/**
 * User: solar
 * Date: 28.04.14
 * Time: 8:31
 */
public class ModelTools {
  public static Func compile(Ensemble<ObliviousTree> input) {
    final BFGrid grid = input.size() > 0 ? input.models[0].grid() : null;
    final TObjectDoubleHashMap<Set<BFGrid.BinaryFeature>> scores = new TObjectDoubleHashMap<Set<BFGrid.BinaryFeature>>();
    for (int i = 0; i < input.size(); i++) {
      final ObliviousTree model = input.models[i];
      final List<BFGrid.BinaryFeature> features = model.features();
      for (int b = 0; b < model.values().length; b++) {
        final Set<BFGrid.BinaryFeature> currentSet = new HashSet<BFGrid.BinaryFeature>();
        final double[] values = model.values();
        double value = 0;
        int bitsB = MathTools.bits(b);
        for (int a = 0; a < values.length; a++) {
          int bitsA = MathTools.bits(a);
          value += MathTools.bits(a & b) >= bitsA ? (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * values[a] : 0;
        }
        for (int f = 0; f < features.size(); f++) {
          if ((b >> f & 1) != 0) {
            currentSet.add(features.get(features.size() - f - 1));
          }
        }
        scores.adjustOrPutValue(currentSet, value, value);
      }
    }
    return new Func.Stub() {
      byte[] binary = new byte[grid.rows()];
      @Override
      public synchronized double value(Vec x) {
        if (grid == null)
          return 0.;

        grid.binarize(x, binary);
        final double[] result = new double[]{0.};
        scores.forEachEntry(new TObjectDoubleProcedure<Set<BFGrid.BinaryFeature>>() {
          @Override
          public boolean execute(Set<BFGrid.BinaryFeature> a, double b) {
            for (BFGrid.BinaryFeature bf : a) {
              if (!bf.value(binary))
                return true;
            }
            result[0] += b;
            return true;
          }
        });
        return result[0];
      }

      @Override
      public int dim() {
        return grid.rows();
      }
    };
  }
}
