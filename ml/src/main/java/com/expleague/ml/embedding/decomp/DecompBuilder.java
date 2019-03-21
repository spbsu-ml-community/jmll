package com.expleague.ml.embedding.decomp;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.CoocBasedBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import gnu.trove.list.array.TIntArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class DecompBuilder extends CoocBasedBuilder {
  private static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  private double xMax = 10;
  private double alpha = 0.75;
  private int symDim = 50;
  private int skewDim = 10;
  private FastRandom rng = new FastRandom();

  public DecompBuilder xMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public DecompBuilder alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public DecompBuilder dimSym(int dim) {
    this.symDim = dim;
    return this;
  }

  public DecompBuilder dimSkew(int dim) {
    this.skewDim = dim;
    return this;
  }

  public DecompBuilder seed(long seed) {
    rng = new FastRandom(seed);
    return this;
  }

  private double weightingFunc(double x) {
    return x < xMax ? Math.pow((x / xMax), alpha) : 1;
  }

  @Override
  public Embedding<CharSeq> fit() {
    final int size = dict().size();
    final Mx symDecomp = new VecBasedMx(size, symDim);
    final Mx skewsymDecomp = new VecBasedMx(size, skewDim);
    final Vec bias = new ArrayVec(size);
    for (int i = 0; i < size; i++) {
      bias.set(i, initializeValue(symDim));
      for (int j = 0; j < symDim; j++) {
        symDecomp.set(i, j, initializeValue(symDim));
      }
      for (int j = 0; j < skewDim; j++) {
        skewsymDecomp.set(i, j, initializeValue(skewDim));
      }
    }

    final Mx softMaxSym = new VecBasedMx(symDecomp.rows(), symDecomp.columns());
    final Mx softMaxSkewsym = new VecBasedMx(skewsymDecomp.rows(), skewsymDecomp.columns());
    final Vec softMaxBias = new ArrayVec(bias.dim());
    VecTools.fill(softMaxSym, 1);
    VecTools.fill(softMaxSkewsym, 1);
    VecTools.fill(softMaxBias, 1);

    final TIntArrayList order = new TIntArrayList(IntStream.range(0, size).toArray());
    rng = new FastRandom();
    for (int iter = 0; iter < T(); iter++) {
      Interval.start();
      order.shuffle(rng);
      final ScoreCalculator scoreCalculator = new ScoreCalculator(size);
      IntStream.range(0, size).parallel().map(order::get).forEach(i -> {
        final Vec sym_i = symDecomp.row(i);
        final Vec skew_i = skewsymDecomp.row(i);
        final Vec softMaxSym_i = softMaxSym.row(i);
        final Vec softMaxSkew_i = softMaxSkewsym.row(i);
        cooc(i, (j, X_ij) -> {
          final Vec sym_j = symDecomp.row(j);
          final Vec skew_j = skewsymDecomp.row(j);
          final Vec softMaxSym_j = softMaxSym.row(j);
          final Vec softMaxSkew_j = softMaxSkewsym.row(j);
          final double b_i = bias.get(i);
          final double b_j = bias.get(j);

          double asum = VecTools.multiply(sym_i, sym_j);
          double bsum = VecTools.multiply(skew_i, skew_j);
          final int sign = i > j ? -1 : 1;
          final double minfo = Math.log(X_ij);
          final double diff = b_i + b_j + asum + sign * bsum - minfo;
          final double weight = weightingFunc(X_ij);
          final double biasStep = weight * diff;
          scoreCalculator.adjust(i, j, weight, 0.5 * weight * MathTools.sqr(diff));

          update(sym_i, softMaxSym_i, sym_j, softMaxSym_j, diff * weight);
          update(skew_i, softMaxSkew_i, skew_j, softMaxSkew_j, diff * weight * sign);
          bias.adjust(i, -step() * biasStep / Math.sqrt(softMaxBias.get(i)));
          softMaxBias.adjust(i, biasStep * biasStep);
          bias.adjust(j, -step() * biasStep / Math.sqrt(softMaxBias.get(j)));
          softMaxBias.adjust(j, biasStep * biasStep);
        });
      });

      project(skewsymDecomp);
      log.info("Iteration: " + iter + ", score: " + scoreCalculator.gloveScore() + ", time: " + Interval.time());
//      Interval.stopAndPrint("Iteration: " + iter + ", score: " + scoreCalculator.gloveScore());
    }

    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < dict().size(); i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, symDecomp.row(i));
    }

//    try (final BufferedWriter writer = Files.newBufferedWriter(Paths.get("/Users/solar/temp/skewsym.txt"))) {
//      for (int i = 0; i < dict().size(); i++) {
//        writer.write(dict().get(i).toString());
//        writer.write('\t');
//        writer.write(MathTools.CONVERSION.convert(skewsymDecomp.row(i), CharSequence.class).toString());
//        writer.write('\n');
//      }
//    }
//    catch (IOException e) {
//      throw new RuntimeException(e);
//    }

    return new EmbeddingImpl<>(mapping);
  }

  private void project(Vec vec) {
    final int dim = vec.dim();
    for (int i = 0; i < dim; i++) {
      final double v = vec.get(i);
      double lambda = 1e-2;
      if (v < -lambda) {
        vec.set(i, v + lambda);
      }
      else if (v > lambda) {
        vec.set(i, v - lambda);
      }
      else {
        vec.set(i, 0.);
      }
    }
  }

  private void update(Vec x_i, Vec softMaxD_i, Vec x_j, Vec softMaxD_j, double step) {
    IntStream.range(0, x_i.dim()).forEach(id -> {
      final double dx_i = x_j.get(id) * step;
      final double dx_j = x_i.get(id) * step;
      final double maxL_i = softMaxD_i.get(id);
      final double maxL_j = softMaxD_j.get(id);
      x_i.adjust(id, -step() * dx_i / Math.sqrt(maxL_i));
      x_j.adjust(id, -step() * dx_j / Math.sqrt(maxL_j));
      softMaxD_i.set(id, maxL_i + MathTools.sqr(dx_i));
      softMaxD_j.set(id, maxL_j + MathTools.sqr(dx_j));
    });
  }

  private double initializeValue(int dim) {
    return (Math.random() - 0.5) / dim;
  }
}
