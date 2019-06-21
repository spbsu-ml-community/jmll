package com.expleague.ml.embedding.kmeans;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static com.expleague.commons.math.vectors.VecTools.*;

@SuppressWarnings("Duplicates")
public class SoftMaxSymmetricBuilder extends EmbeddingBuilderBase {
  private int dim = 50;
  private Mx result; // смещение всех векторов относительно центроидов, для центроидов - нули.

  private FastRandom rng = new FastRandom(100500);

  public SoftMaxSymmetricBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  @Override
  protected Embedding<CharSeq> fit() {
    initialize();

    log.info("Started scanning corpus." + this.path);
    for (int t = 0; t < T(); t++) {
      System.out.println();
      log.info("\nEpoch " + t);
      try (final LongStream stream = positionsStream()) {
        stream.forEach(tuple -> {
          move(unpackA(tuple), unpackB(tuple), unpackWeight(tuple));
        });
      }
      catch (IOException e) {
        throw new RuntimeException("Error in source function occured\n" + e.getMessage());
      }
    }
    System.out.println();
    final Map<CharSeq, Vec> result = new HashMap<>();
    for (int i = 0; i < dict().size(); i++) {
      result.put(dict().get(i), this.result.row(i));
    }
    return new EmbeddingImpl<>(result);
  }

  private double score = 0;
  private static double G = 0.99999;
  private long idx = 0;
  private void move(int i, int j, double weight) {
//    if ((-1 - 1000 * Math.log(rng.nextDouble())) > i)
//      return;
    if (i == j)
      return;
    if (rng.nextDouble() < 0.5)
      return;
//    weight *= Math.min(1, 1e-4 / p(i));
    final int wordsCount = dict().size();
    final Vec v_i = result.row(wordsCount + i);
    final Vec v_j = result.row(j);

    double product = multiply(v_i, v_j);
    final Vec grad = new ArrayVec(dim);
    final Vec weights = MxTools.multiply(new VecBasedMx(dim, result.vec().sub(0, wordsCount * dim)), v_i);
    double maxLogWeight = max(weights);
    final double correctW;
    { // normalization for better arithmetic stability
      IntStream.range(0, weights.dim()).forEach(idx -> weights.set(idx, idx == i ? 0 : Math.exp(weights.get(idx) - maxLogWeight) * p(idx)));
      correctW = p(j) * Math.exp(product - maxLogWeight);
    }
    double denom = sum(weights);

    for (int k = 0; k < wordsCount; k++) {
      if (k == i || k == j)
        continue;
      final double pointGrad = -weights.get(k) / denom;
      if (Math.abs(pointGrad) > 1e-4) {
        final Vec point_k = result.row(k);
        incscale(grad, point_k, pointGrad);
        adaStep(k, point_k, v_i, step(), weight * pointGrad);
      }
    }
    incscale(grad, v_j, (1 - correctW / denom));
    adaStep(j, v_j, v_i, step(), weight * (1 - correctW / denom));
    adaStep(i, v_i, grad, step(), weight);


    final double score = (Math.log(p(j)) + product - maxLogWeight - Math.log(denom));
    this.score = this.score * G + score;

    if ((++idx % 10000) == 0) {
      System.out.print("\r" + idx + " score: " + CharSeqTools.ppDouble(score) + " total score: " + CharSeqTools.ppDouble(this.score / (1 - Math.pow(G, idx)) * (1 - G)));
    }
  }

  private List<Vec> accum = new ArrayList<>();
  private void adaStep(int index, Vec x, Vec grad, double step, double scale) {
    if (Math.abs(scale) < 1e-8)
      return;

    while (accum.size() <= index) {
      accum.add(VecTools.fill(new ArrayVec(x.dim()), 1));
    }
    final Vec accum = this.accum.get(index);
    double len = IntStream.range(0, x.dim()).mapToDouble(i -> {
      final double increment = step * scale * grad.get(i) / Math.sqrt(accum.get(i));
      x.adjust(i, increment);
      accum.adjust(i, MathTools.sqr(scale * grad.get(i)));
      return increment * increment;
    }).sum();
    VecTools.normalizeL2(x);
  }

  private void initialize() {
    final int voc_size = dict().size();
    final FastRandom rng = new FastRandom();
    result = new VecBasedMx(voc_size * 2, dim);
    fillUniform(result, rng);
    for (int i = 0; i < result.rows(); i++) {
      normalizeL2(result.row(i));
    }
  }
}
