package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;
import com.expleague.ml.optimization.impl.FullGradientDescent;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class PNFATest {
  private final FastRandom random = new FastRandom(1);
  private final static double EPS = 1e-6;
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

  @Test
  public void testNot() {
    final int stateCount = 4;
    PNFA<WeightedL2> pnfa = new PNFA<>(
        stateCount, 3,4, 0.0, 20, random,
        new AdamDescent(random, 50, 4, 0.0004 ),
        new FullGradientDescent(random, 1, 5),
        1
    );

    final int TRAIN_SIZE = 10000;
    List<IntSeq> train = new ArrayList<>();
    int[] labels = new int[TRAIN_SIZE];

    for (int i = 0; i < TRAIN_SIZE; i++) {
      int len = random.nextInt(80) + 10;
      int[] seq = new int[len];
      for (int j = 0; j < len; j++) {
        int z = random.nextInt(j == len - 1 ? 3 : 4);
        seq[j] = z;
      }
      train.add(new IntSeq(seq));

      int s = seq[len - 1] - 1;
      if (seq[len - 2] == 3) {
        s = -s;
      }
      labels[i] = s + 1;
    }

    final Vec target = new ArrayVec(TRAIN_SIZE * 3);
    for (int i = 0; i < TRAIN_SIZE; i++) {
      target.set(i * 3 + labels[i], 1);
    }

    Function<Seq<Integer>, Vec> model = pnfa.fit(new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return train.get(i);
      }

      @Override
      public int length() {
        return TRAIN_SIZE;
      }

      @Override
      public Class<Seq<Integer>> elementType() {
        return null;
      }
    }, new WeightedL2(target, null));

    int acc = 0;
    for (int i = 0; i < TRAIN_SIZE; i++) {

      int z = VecTools.argmax(model.apply(train.get(i)));
      if (z == labels[i]) acc++;
      else {
        int keke = 2;
      }
    }
    System.out.println(acc + " " + 1.0 * acc / TRAIN_SIZE);


    printMx(PNFA.getWeightMx(((PNFA.PNFAModel) model).getParams(), stateCount, 20, 0, 0));
    System.out.println("=========");
    printMx(PNFA.getWeightMx(((PNFA.PNFAModel) model).getParams(), stateCount, 20, 0, 1));
    System.out.println("=========");
    printMx(PNFA.getWeightMx(((PNFA.PNFAModel) model).getParams(), stateCount, 20, 0,2));
    System.out.println("=========");
    printMx(PNFA.getWeightMx(((PNFA.PNFAModel) model).getParams(), stateCount, 20, 0,3));
  }

  private void printMx(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        System.out.printf("%.3f ", mx.get(i, j));
      }
      System.out.println();
    }
  }
}