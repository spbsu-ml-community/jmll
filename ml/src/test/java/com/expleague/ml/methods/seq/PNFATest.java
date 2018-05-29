package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.seq.param.*;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.SAGADescent;
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
    public Vec optimize(FuncEnsemble<? extends FuncC1> func, RegularizerFunc reg, Vec x0) {
      PNFAItemVecRegression model = (PNFAItemVecRegression) func.models[0];

      final double value = model.trans(x0).get(0);
      final Vec grad = model.gradient(x0);

      for (int i = 0; i < x0.dim() - stateCount * stateDim; i++) {
        x0.adjust(i, EPS);
        final double newValue = model.trans(x0).get(0);
        System.err.println(i + " " + grad.get(i));
        double expected = (newValue - value) / EPS;
        if (Math.abs(grad.get(i) - expected) > 1e-3) {
          model.trans(x0).get(0);
          model.gradient(x0);
          assertEquals(expected, grad.get(i), 1e-3);
        }
        x0.adjust(i, -EPS);
      }
      return x0;
    }
  };

  @Test
  public void testPNFAGradient() {
    IntAlphabet alphabet = new IntAlphabet(10);
//    PNFA<WeightedL2> pnfaRegressor = new PNFA<>(
//        stateCount, stateDim,10, 0.0, 1, random, weightOptimize, valueOptimize, 1
//    );
    final double addToDiag = 10;

    BettaParametrization bettaParametrization = new BettaMxParametrization(addToDiag);
    WeightParametrization weightParametrization = new WeightSquareParametrization(bettaParametrization);

    PNFARegressor<Integer, WeightedL2> pnfaRegressor = new PNFARegressor<>(
        stateCount, stateCount, stateDim, alphabet, 0.0001, 0.001, addToDiag, 0.1, random, weightOptimize, bettaParametrization, weightParametrization);

    pnfaRegressor.fit(new DataSet.Stub<Seq<Integer>>(null) {
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
  public void testSimple() {
    final int stateCount = 2;
    final IntAlphabet alphabet = new IntAlphabet(2);
    int diag = 5;
    BettaParametrization bettaParametrization = new BettaTwoVecParametrization(diag);
    WeightParametrization weightParametrization = new WeightSquareParametrization(bettaParametrization);

    final PNFARegressor<Integer, WeightedL2> pnfaRegressor = new PNFARegressor<>(
        stateCount,
        stateCount,
        stateCount,
        alphabet,
        1e-5,
        1e-3,
        diag,
        0.0,
        random,
        new SAGADescent(0.01, 100000, random, System.out),
        //        new AdamDescent(random, 50, 4, 0.0001)
//        new OnlineDescent(0.001, random)
//        new FullGradientDescent(random, 0.3, 1000)
        bettaParametrization,
        weightParametrization
    );

    final int TRAIN_SIZE = 1000;
    List<IntSeq> train = new ArrayList<>();
    int[] labels = new int[TRAIN_SIZE];

    for (int i = 0; i < TRAIN_SIZE; i++) {
      int len = random.nextInt(80) + 10;
      IntSeqBuilder builder = new IntSeqBuilder();
      for (int j = 0; j < len; j++) {
        builder.append(random.nextInt(2));
      }
      IntSeq seq = builder.build();
      train.add(seq);

      labels[i] = seq.intAt(seq.length() - 1);
    }

    final Vec target = new ArrayVec(TRAIN_SIZE * 2);
    for (int i = 0; i < TRAIN_SIZE; i++) {
      target.set(i * 2 + labels[i], 1);
    }

    Function<Seq<Integer>, Vec> model = pnfaRegressor.fit(new DataSet.Stub<Seq<Integer>>(null) {
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


    printMx(weightParametrization.getMx(((PNFAModel) model).getParams(), 0, stateCount));
    System.out.println("=========");
    printMx(weightParametrization.getMx(((PNFAModel) model).getParams(), 1, stateCount));
  }

  @Test
  public void testNot() {
    final int stateCount = 3;
    final IntAlphabet alphabet = new IntAlphabet(3);
    int diag = 10;
//    final PNFA<WeightedL2> pnfaRegressor = new PNFA<>(
//        stateCount, 2, alphabet.size(), 0, diag, random,
////        new SAGADescent(0.3, 10000, random, 1),
//        new AdamDescent(random, 50, 4, 0.0004 ),
//        new FullGradientDescent(random, 0.3, 10),
//        1
//    );
    BettaParametrization bettaParametrization = new BettaTwoVecParametrization(diag);
    WeightParametrization weightParametrization = new WeightSquareParametrization(bettaParametrization);

    final PNFARegressor<Integer, WeightedL2> pnfaRegressor = new PNFARegressor<>(
        2, stateCount, 2, alphabet, 1e-5, 1e-3, diag, 0.5, random,
        new SAGADescent(0.001, 1000000, random, System.out),
        //        new AdamDescent(random, 50, 4, 0.001)
        bettaParametrization, weightParametrization);

    final int TRAIN_SIZE = 10000;
    List<IntSeq> train = new ArrayList<>();
    int[] labels = new int[TRAIN_SIZE];

    for (int i = 0; i < TRAIN_SIZE; i++) {
      int len = 2;//random.nextInt(80) + 10;
      IntSeqBuilder builder = new IntSeqBuilder();
      for (int j = 0; j < len; j++) {
        builder.append(random.nextInt(j == len - 1 ? 2 : 3));
      }
      IntSeq seq = builder.build();
      train.add(seq);

      int s = 2 * seq.intAt(seq.length() - 1) - 1;
      if (seq.intAt(seq.length() - 2) == 2) {
        s = -s;
      }
      labels[i] = (s + 1) / 2;
    }

    final Vec target = new ArrayVec(TRAIN_SIZE * 2);
    for (int i = 0; i < TRAIN_SIZE; i++) {
      target.set(i * 2 + labels[i], 1);
    }

    Function<Seq<Integer>, Vec> model = pnfaRegressor.fit(new DataSet.Stub<Seq<Integer>>(null) {
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


    printMx(weightParametrization.getMx(((PNFAModel) model).getParams(), 0, stateCount));
    System.out.println("=========");
    printMx(weightParametrization.getMx(((PNFAModel) model).getParams(), 1, stateCount));
    System.out.println("=========");
    printMx(weightParametrization.getMx(((PNFAModel) model).getParams(), 2, stateCount));
//    System.out.println("=========");
//    printMx(PNFAItemVecRegression.weightMx(((PNFARegressor.PNFAModel) model).getParams(), 3, stateCount, diag));
  }

  @Test
  public void testRE1() {
//    final int stateCount = 3;
//    PNFARegressor<Character, WeightedL2> pnfaRegressor = new PNFARegressor<>(
//        stateCount, 3, CharSeqTools.BASE64ALPHA, 0.1, 20, random,
//        new SAGADescent(0.3, 10000, random, 1),
//        new FullGradientDescent(random, 0.3, 10),
//        1
//    );
//
//    final int TRAIN_SIZE = 10000;
//    final List<CharSeq> train = new ArrayList<>();
//    final Pattern pattern = Pattern.compile("AB");
//    final int[] labels = new int[TRAIN_SIZE];
//
//    for (int i = 0; i < TRAIN_SIZE; i++) {
//      final String sample = random.nextBase64String(random.nextInt(80) + 10);
//      train.add(CharSeq.create(sample));
//      labels[i] = pattern.asPredicate().test(sample) ? 1 : -1;
//    }
//
//    final Vec target = new ArrayVec(TRAIN_SIZE);
//    for (int i = 0; i < TRAIN_SIZE; i++) {
//      target.set(i, labels[i]);
//    }
//
//    Function<Seq<Character>, Vec> model = pnfaRegressor.fit(new DataSet.Stub<Seq<Character>>(null) {
//      @Override
//      public CharSeq at(int i) {
//        return train.get(i);
//      }
//
//      @Override
//      public int length() {
//        return TRAIN_SIZE;
//      }
//
//      @Override
//      public Class<CharSeq> elementType() {
//        return CharSeq.class;
//      }
//    }, new WeightedL2(target, null));
//
//    int acc = 0;
//    for (int i = 0; i < TRAIN_SIZE; i++) {
//
//      int z = VecTools.argmax(model.apply(train.get(i)));
//      if (z == labels[i]) acc++;
//      else {
//        int keke = 2;
//      }
//    }
//    System.out.println(acc + " " + 1.0 * acc / TRAIN_SIZE);
//
//
//    printMx(PNFAItemVecRegression.weightMx(((PNFARegressor.PNFAModel) model).getParams(), 0, 20, 0));
//    System.out.println("=========");
//    printMx(PNFAItemVecRegression.weightMx(((PNFARegressor.PNFAModel) model).getParams(), 1, 20, 0));
//    System.out.println("=========");
//    printMx(PNFAItemVecRegression.weightMx(((PNFARegressor.PNFAModel) model).getParams(), 2, 20, 0));
//    System.out.println("=========");
//    printMx(PNFAItemVecRegression.weightMx(((PNFARegressor.PNFAModel) model).getParams(), 3, 20, 0));
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