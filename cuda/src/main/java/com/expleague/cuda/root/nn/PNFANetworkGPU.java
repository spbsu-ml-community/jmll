package com.expleague.cuda.root.nn;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.commons.util.ArrayTools;
import com.expleague.cuda.data.impl.DoubleVector;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.seq.automaton.DFA;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by hrundelb on 20.08.17.
 */
public class PNFANetworkGPU<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int stateCount;
  private final Alphabet<T> alphabet;
  private final Random random;
  private final double step;
  private final int iterationsCount;


  private final DoubleVector lastValuesGradSeqIdGPU;
  private final DoubleVector seqBetaGradGPU;
  private final DoubleVector totalBetaGradGPU;
  private final DoubleVector lastBetaGradGPU;
  private final DoubleVector gradBetaGradGPU;
  private final DoubleVector valuesGradGPU;
  private final DoubleVector gradValuesGradGPU;

  public PNFANetworkGPU(final Alphabet<T> alphabet,
                         final int stateCount,
                         final Random random,
                         final double step,
                         final int iterationsCount) {
    this.stateCount = stateCount;
    this.alphabet = alphabet;
    this.random = random;
    this.step = step;
    this.iterationsCount = iterationsCount;

    lastValuesGradSeqIdGPU = new DoubleVector(new double[stateCount]);
    seqBetaGradGPU = new DoubleVector(new double[stateCount * (stateCount - 1)]);
    totalBetaGradGPU = new DoubleVector(new double[stateCount * (stateCount - 1)]);
    lastBetaGradGPU = new DoubleVector(new double[stateCount * (stateCount - 1)]);
    gradBetaGradGPU = new DoubleVector(new double[stateCount * (stateCount - 1)]);
    valuesGradGPU = new DoubleVector(new double[stateCount]);
    gradValuesGradGPU = new DoubleVector(new double[stateCount]);
  }

  @Override
  public Computable<Seq<T>, Vec> fit(DataSet<Seq<T>> learn, Loss loss) {
    final PNFAParamsGPU<T> params = new PNFAParamsGPU<>(random, stateCount, alphabet);

    int maxLen = 0;
    for (int i = 0; i < learn.length(); i++) {
      maxLen = Math.max(maxLen, learn.at(i).length());
    }

    final Vec[] lastValuesGrad = new Vec[learn.length()];
    final Mx[][] lastBetaGrad = new Mx[learn.length()][];
    final int[] lastAlphaGradChanged = new int[alphabet.size()];
    final int[][] seqAlphabet = new int[learn.length()][];

    final Mx[] totalBetaGrad = new Mx[alphabet.size()];
    for (int i = 0; i < alphabet.size(); i++) {
      totalBetaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
    }
    //final Vec totalValuesGrad = new ArrayVec(stateCount);
    DoubleVector totalValuesGradGPU = new DoubleVector(new double[stateCount]);

    Vec valuesGrad = new ArrayVec(stateCount);
    final Mx[] betaGrad = new Mx[alphabet.size()];
    for (int i = 0; i < alphabet.size(); i++) {
      betaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
    }

    for (int i = 0; i < learn.length(); i++) {
      final Seq<T> seq = learn.at(i);
      List<Integer> charsList = new ArrayList<>(seq.length());
      for (int j = 0; j < seq.length(); j++) {
        charsList.add(alphabet.index(seq.at(j)));
      }
      seqAlphabet[i] = charsList.stream().sorted().distinct().mapToInt(x -> x).toArray();
      lastBetaGrad[i] = new Mx[seqAlphabet[i].length];
      final PNFAParamsGPU.PNFAParamsGradGPU grad = params.calcSeqGrad(seq, seqAlphabet[i], loss
          .target().at(i));
      lastBetaGrad[i] = grad.getBetaGrad();
      lastValuesGrad[i] = grad.getValuesGrad();

      //VecTools.append(totalValuesGrad, grad.getValuesGrad());
      //JCublasHelper.append(totalValuesGradGPU, new DoubleVector(grad.getValuesGrad().toArray()));
      for (int j = 0; j < seqAlphabet[i].length; j++) {
        final int c = seqAlphabet[i][j];
        VecTools.append(totalBetaGrad[c], grad.getBetaGrad()[j]);
      }
    }

    for (int c = 0; c < alphabet.size(); c++) {
      VecTools.incscale(betaGrad[c], totalBetaGrad[c], 1.0 / learn.length());
    }

    VecTools.incscale(valuesGrad, new ArrayVec(totalValuesGradGPU.get()), 1.0 / learn.length());

    double curCost = getCost(learn, loss, params);
    long start = System.nanoTime();

    final Mx[] seqBetaGrad = new Mx[maxLen];
    Vec seqValuesGrad = new ArrayVec(stateCount);

    for (int iter = 1; iter < iterationsCount; iter++) {
      final int seqId = random.nextInt(learn.length());
      final Seq<T> seq = learn.at(seqId);

      final PNFAParamsGPU.PNFAParamsGradGPU grad = params.calcSeqGrad(seq, seqAlphabet[seqId],
          loss.target().at(seqId));

      valuesGrad = extracted2(learn.length(), lastValuesGrad[seqId], lastBetaGrad[seqId],
          lastAlphaGradChanged, seqAlphabet[seqId], totalBetaGrad, totalValuesGradGPU,
          valuesGrad, betaGrad, seqBetaGrad, iter, grad);
      lastValuesGrad[seqId] = grad.getValuesGrad();

      // do not apply update to all the characters
      params.updateParams(seqBetaGrad, valuesGrad, -step, seqAlphabet[seqId]);

      if (iter % (5 * learn.length()) == 0) {
        long cur = System.nanoTime();

        for (int seqId1 = 0; seqId1 < learn.length(); seqId1++) {
          for (int i = 0; i < seqAlphabet[seqId1].length; i++) {
            final int a = seqAlphabet[seqId1][i];
            seqBetaGrad[i] = betaGrad[a];
            VecTools.fill(seqBetaGrad[i], 0);
            // character a was not counted during last (iter - lastAlphaGradChanged[a] - 1) iterations as part of
            // the average sum of derivatives
            VecTools.incscale(seqBetaGrad[i], totalBetaGrad[a], (double) (iter - lastAlphaGradChanged[a]) / learn.length());
            lastAlphaGradChanged[a] = iter;
          }
          VecTools.fill(valuesGrad, 0);
          params.updateParams(seqBetaGrad, valuesGrad, -step, seqAlphabet[seqId1]);
        }
        final double newCost = getCost(learn, loss, params);
        System.out.printf("Iteration %d, cost=%.6f\n", iter, newCost);
        //  System.out.printf("Iterations elapsed  %d, cost=%.6f, 100 iterations passed in %.2f minutes\n", iter, curCost, (cur - start) / 60e9);
        start = cur;
        if (newCost > curCost && iter != 0) {
          System.out.printf("Iterations elapsed %d, cost=%.6f\n", iter, curCost);
          break;
        }
        System.out.flush();
        curCost= newCost;
      }
    }
    DFA<T> result = new DFA<>(alphabet);
    for (int i = 1; i < stateCount; i++) result.createNewState();

    for (int i = 0; i < alphabet.size(); i++) {
      for (int j = 0; j < stateCount; j++) {
        int id = ArrayTools.max(params.getW()[i].row(j).toArray());
        result.addTransition(j, id, alphabet.getT(alphabet.get(i)));
      }
    }
//    return (seq) -> new SingleValueVec(params.getValues().get(result.run(seq))); //new SingleValueVec(params.getSeqValue(seq));

    lastValuesGradSeqIdGPU.destroy();
    seqBetaGradGPU.destroy();
    totalBetaGradGPU.destroy();
    lastBetaGradGPU.destroy();
    gradBetaGradGPU.destroy();
    valuesGradGPU.destroy();
    gradValuesGradGPU.destroy();

    return (seq) -> new SingleValueVec(params.getSeqValue(seq));
  }


  private Vec extracted2(final int learnlen, Vec lastValuesGradSeqId, Mx[] lastBetaGrad, int[]
      lastAlphaGradChanged, int[] seqAlphabet, Mx[] totalBetaGrad, DoubleVector
      totalValuesGradGPU, Vec valuesGrad, Mx[] betaGrad, Mx[] seqBetaGrad, int iter, PNFAParamsGPU
      .PNFAParamsGradGPU grad) {

    // Host -> Device
    lastValuesGradSeqIdGPU.reset(lastValuesGradSeqId.toArray());

    for (int i = 0; i < seqAlphabet.length; i++) {
      final int a = seqAlphabet[i];

      // Host -> Device
      seqBetaGrad[i] = betaGrad[a];
      //VecTools.fill(seqBetaGrad[i], 0);
      seqBetaGradGPU.reproduce();
      totalBetaGradGPU.reset(totalBetaGrad[a].toArray());
      lastBetaGradGPU.reset(lastBetaGrad[i].toArray());
      gradBetaGradGPU.reset(grad.getBetaGrad()[i].toArray());

      // character a was not counted during last (iter - lastAlphaGradChanged[a] - 1) iterations
      // as part of
      // the average sum of derivatives
      double scale = (double) (iter - lastAlphaGradChanged[a]) / learnlen;
      //VecTools.incscale(seqBetaGrad[i], totalBetaGrad[a], scale);
      //JCublasHelper.incscale(seqBetaGradGPU, totalBetaGradGPU, scale);


      //VecTools.append(seqBetaGrad[i], grad.getBetaGrad()[i]);
      //VecTools.incscale(seqBetaGrad[i], lastBetaGrad[i], -1);
      //JCublasHelper.append(seqBetaGradGPU, gradBetaGradGPU);
      //JCublasHelper.incscale(seqBetaGradGPU, lastBetaGradGPU, -1);

      //VecTools.incscale(totalBetaGrad[a], lastBetaGrad[i], -1);
      //VecTools.append(totalBetaGrad[a], grad.getBetaGrad()[i]);
      //JCublasHelper.incscale(totalBetaGradGPU, lastBetaGradGPU, -1);
      //JCublasHelper.append(totalBetaGradGPU, gradBetaGradGPU);

      // do nothing
      lastAlphaGradChanged[a] = iter;
      lastBetaGrad[i] = grad.getBetaGrad()[i];
      //Device -> Host
      seqBetaGrad[i] = new VecBasedMx(seqBetaGrad[i].columns(), new ArrayVec(seqBetaGradGPU.get()));
      totalBetaGrad[a] = new VecBasedMx(totalBetaGrad[a].columns(), new ArrayVec(totalBetaGradGPU
          .get()));
    }

    //VecTools.fill(valuesGrad, 0);
    // Host -> Device
    valuesGradGPU.reproduce();
    gradValuesGradGPU.reset(grad.getValuesGrad().toArray());

    //VecTools.append(valuesGrad, grad.getValuesGrad());
    //VecTools.incscale(valuesGrad, lastValuesGradSeqId, -1);
    //VecTools.incscale(valuesGrad, totalValuesGrad, 1.0 / learnlen);
    //JCublasHelper.append(valuesGradGPU, gradValuesGradGPU);
    //JCublasHelper.incscale(valuesGradGPU, lastValuesGradSeqIdGPU, -1);
    //JCublasHelper.incscale(valuesGradGPU, totalValuesGradGPU, 1.0 / learnlen);

    //VecTools.incscale(totalValuesGrad, lastValuesGradSeqId, -1);
    //VecTools.append(totalValuesGrad, grad.getValuesGrad());
    //JCublasHelper.incscale(totalValuesGradGPU, lastValuesGradSeqIdGPU, -1);
    //JCublasHelper.append(totalValuesGradGPU, gradValuesGradGPU);

    // Device -> Host
    valuesGrad = new ArrayVec(valuesGradGPU.get());
    return valuesGrad;

  }

  private double getCost(final DataSet<Seq<T>> learn, final Loss loss, final PNFAParamsGPU<T>
      params) {
    double cost = 0;
    for (int i = 0; i < learn.length(); i++) {
      cost += MathTools.sqr(params.getSeqValue(learn.at(i)) - loss.target().at(i));
    }
    return cost / learn.length();
  }
}