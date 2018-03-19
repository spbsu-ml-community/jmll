package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.optimization.Optimize;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class PNFATest {
  private final FastRandom random = new FastRandom(1);
  private final static double EPS = 1e-5;
  private final int stateCount = 4;
  private final int stateDim = 3;

  private final Optimize<FuncEnsemble<? extends FuncC1>> weightOptimize = new Optimize<FuncEnsemble<? extends FuncC1>>() {
    @Override
    public Vec optimize(FuncEnsemble func) {
      assertTrue(false);
      return null;
    }

    @Override
    public Vec optimize(FuncEnsemble<? extends FuncC1> func, Vec x0) {
      final double value = func.models[0].trans(x0).get(0);
      final Vec grad = func.models[0].gradient(x0);

      for (int i = 0; i < x0.dim() - stateCount * stateDim; i++) {
        x0.adjust(i, EPS);
        final double newValue = func.models[0].trans(x0).get(0);
        System.err.println(i + " " + grad.get(i));
        assertEquals(grad.get(i), (newValue - value) / EPS, 1e-3);
        x0.adjust(i, -EPS);
      }
      return x0;
    }
  };

  private final Optimize<FuncEnsemble<? extends FuncC1>> valueOptimize = new Optimize<FuncEnsemble<? extends FuncC1>>() {
    @Override
    public Vec optimize(FuncEnsemble func) {
      assertTrue(false);
      return null;
    }

    @Override
    public Vec optimize(FuncEnsemble<? extends FuncC1> func, Vec x0) {
      final double value = func.models[0].trans(x0).get(0);
      final Vec grad = func.models[0].gradient(x0);

      for (int i = x0.dim() - stateCount * stateDim; i < x0.dim(); i++) {
        x0.adjust(i, EPS);
        final double newValue = func.models[0].trans(x0).get(0);
        assertEquals(grad.get(i), (newValue - value) / EPS, 1e-3);
        x0.adjust(i, -EPS);
      }
      return x0;
    }
  };

  @Test
  public void testPNFAGradient() {
    PNFA<WeightedL2> pnfa = new PNFA<>(
        stateCount, stateDim,10, 0.0, 1, random, weightOptimize, valueOptimize, 1
    );

    pnfa.fit(new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return new IntSeq(0, 1, 2, 3, 1, 2, 3, 7, 8, 9);
      }

      @Override
      public int length() {
        return 2;
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    }, new WeightedL2(new ArrayVec(-1.0, 1.0, -23, 32, 44, 2), null));
  }
}