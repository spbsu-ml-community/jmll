package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.SeqOptimization;
import com.spbsu.ml.methods.seq.nn.PNFANetworkGD;
import com.spbsu.ml.methods.seq.nn.PNFANetworkSAGA;
import com.spbsu.ml.methods.seq.nn.PNFANetworkSGD;
import com.spbsu.ml.methods.seq.nn.PNFAParams;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TestMethods {
  private final static Random random = new Random(239);
  private DataSet<Seq<Integer>> trainData;
  private DataSet<Seq<Integer>> testData;
  private Vec trainTarget;
  private Vec testTarget;

  private final int alphabetSize = 10;
  @Before
  public void genData() {
    final Function<Seq<Integer>, Integer> classifier = seq -> {
      int cnt = 0;
      for (int i = 0; i < seq.length(); i++) {
        cnt += seq.at(i) >= alphabetSize / 2 ? 1 : 0;
      }
      return 2 * cnt >= seq.length() ? 1 : -1;
    };

    final Pair<DataSet<Seq<Integer>>, Vec> train = generateData(classifier, 2000, 20, 40, 2, alphabetSize);
    trainData = train.first;
    trainTarget = train.second;

    final Pair<DataSet<Seq<Integer>>, Vec> test = generateData(classifier, 2000, 20, 40, 2, alphabetSize);
    testData = test.first;
    testTarget = test.second;
  }

  @Test
  public void testSAGA() {
    SeqOptimization<Integer, L2> net = new PNFANetworkSAGA<>(new IntAlphabet(alphabetSize), 2, random, 0.1, 1000000);

    final Computable<Seq<Integer>, Vec> classifier = net.fit(trainData, new L2(trainTarget, trainData));
    //final Computable<Seq<Integer>, Vec> classifier = new GradientSeqBoosting<>(net, 30, 0.02).fit(trainData, new LLLogit(trainTarget, trainData));
    System.out.println("Saga train accuracy: " + getAccuracy(trainData, trainTarget, classifier));
    System.out.println("Saga test accuracy: " + getAccuracy(testData, testTarget, classifier));
  }

  private Pair<DataSet<Seq<Integer>>, Vec> generateData(Function<Seq<Integer>, Integer> classifier,
                                                        int sampleCount, int minLen, int maxLen, int classCount, int alphabetSize) {
    int[] classSamplesCount = new int[classCount];
    final List<Seq<Integer>> data = new ArrayList<>(sampleCount);
    int[] target = new int[sampleCount];
    while (data.size() < sampleCount) {
      int len = random.nextInt(maxLen - minLen + 1) + minLen;
      int[] sample = new int[len];
      for (int i = 0; i < len; i++) {
        sample[i] = random.nextInt(alphabetSize);
      }
      final IntSeq seq = new IntSeq(sample);
      int clazz = classifier.apply(seq);
      if (classSamplesCount[(clazz + 1) / 2] <= data.size() / classCount) {
        target[data.size()] = clazz;
        data.add(seq);
        classSamplesCount[(clazz + 1) / 2]++;
      }
    }
    DataSet<Seq<Integer>> dataSet = new DataSet.Stub<Seq<Integer>>(null) {
      @Override
      public Seq<Integer> at(int i) {
        return data.get(i);
      }

      @Override
      public int length() {
        return data.size();
      }

      @Override
      public Class elementType() {
        return Seq.class;
      }
    };
    return new Pair<>(dataSet, VecTools.fromIntSeq(new IntSeq(target)));
  }

  @Test
  public void testSeqGrad() {
    final int stateCount = 5;
    PNFAParams<Integer> params = new PNFAParams<>(random, stateCount, new IntAlphabet(6));
    Seq<Integer> seq = new IntSeq(5, 1, 2, 3, 1, 1, 0, 0);
    final int[] seqAlphabet = {0, 1, 2, 3, 5};
    final PNFAParams.PNFAParamsGrad grad = params.calcSeqGrad(seq, seqAlphabet, 0);
    final Mx[] beta = params.getBeta();
    final Vec values = params.getValues();
    final double val = params.getSeqValue(seq);
    final double eps = 1e-6;
    for (int i = 0; i < stateCount; i++) {
      values.adjust(i, eps);
      final double newVal = params.getSeqValue(seq);
      values.adjust(i, -eps);
      assertEquals(2 * val * (newVal - val) / eps, grad.getValuesGrad().get(i), 1e-6);

    }
    for (int i = 0; i < seqAlphabet.length; i++) {
      for (int j = 0; j < beta[0].rows(); j++) {
        for (int k = 0; k < beta[0].columns(); k++) {
          beta[seqAlphabet[i]].adjust(j, k, eps);
          params.updateWeights();
          final double newVal = params.getSeqValue(seq);
          beta[seqAlphabet[i]].adjust(j, k, -eps);
          assertEquals(2 * val * (newVal - val) / eps, grad.getBetaGrad()[i].get(j, k), 1e-6);
        }
      }
    }
  }

  private double getAccuracy(DataSet<Seq<Integer>> data, Vec target, Computable<Seq<Integer>, Vec> computable) {
    int matchCnt = 0;
    for (int i = 0; i < data.length(); i++) {
      final double val = computable.compute(data.at(i)).at(0);
      if (val <= 0 &&  target.at(i) <= 0) {
        matchCnt++;
      } else if (val >= 0 && target.at(i) >= 0) {
        matchCnt++;
      }
    }
    return 1.0 * matchCnt / data.length();
  }

}
