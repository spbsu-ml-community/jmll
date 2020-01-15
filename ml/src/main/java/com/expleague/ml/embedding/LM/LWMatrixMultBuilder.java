package com.expleague.ml.embedding.LM;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.impl.LanguageModelBuiderBase;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectIntMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static java.lang.Float.NaN;
import static java.lang.Float.POSITIVE_INFINITY;
import static java.util.stream.IntStream.range;

public class LWMatrixMultBuilder extends LanguageModelBuiderBase {
  private static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  private static final double MIN_PROBAB = 1e-10;
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 50;
  private FastRandom rng = new FastRandom();
  private boolean regularization = false;
  private Mx contextSymVectors, contextSkewVectors, imageVectors;
  private List<Mx> contextMatrices;
  private Mx C0 = C0();

  public LWMatrixMultBuilder xMax(int xMax) {
    this.xMax = xMax;
    return this;
  }

  public LWMatrixMultBuilder alpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public LWMatrixMultBuilder dim(int dim) {
    this.dim = dim;
    this.C0 = C0();
    return this;
  }

  public LWMatrixMultBuilder seed(long seed) {
    rng = new FastRandom(seed);
    return this;
  }

  public LWMatrixMultBuilder regularization(boolean flag) {
    regularization = flag;
    return this;
  }

  public TObjectIntMap<CharSeq> getWords() {
    return wordsIndex;
  }

  public List<CharSeq> getVocab() {
    return wordsList;
  }

  @Override
  public Embedding<CharSeq> fit() {
    final int vocab_size = dict().size();
    final int texts_number = textsNumber();
    final int window_left = wleft();
    final int window_right = wright();
    contextSymVectors = new VecBasedMx(vocab_size, dim);
    contextSkewVectors = new VecBasedMx(vocab_size, dim);
    contextMatrices = new ArrayList<>(vocab_size);
    imageVectors = new VecBasedMx(vocab_size, dim);

    for (int i = 0; i < vocab_size; i++) {
      Mx mat = new VecBasedMx(dim, dim);
      for (int j = 0; j < dim; j++) {
        contextSymVectors.set(i, j, initializeValue(dim));
        contextSkewVectors.set(i, j, initializeValue(dim));
        imageVectors.set(i, j, initializeValue(dim));
        for (int k = 0; k < dim; k++) {
          mat.set(j, k, initializeValue(dim));
        }
      }
      VecTools.normalizeL2(mat);
      contextMatrices.add(mat);
      VecTools.normalizeL2(contextSymVectors.row(i));
      VecTools.normalizeL2(contextSkewVectors.row(i));
      VecTools.normalizeL2(imageVectors.row(i));
    }


    //checkDerivative(window_left, window_right);

    final TIntArrayList order = new TIntArrayList(range(0, texts_number).toArray());
    rng = new FastRandom();
    for (int iter = 0; iter < T(); iter++) {
      log.info("\nITERATION NUMBER " + iter);
      Interval.start();
      order.shuffle(rng);

      // ОБНОВЛЯЕМ ВЕКТОРА КОНТЕКСТА

      // Перебираем тексты
      range(0, texts_number).parallel().map(order::get).forEach(txt -> {
        final IntSeq text = text(txt);
        // Перебираем слова в тексте по очереди, чтобы их обновить вектора
        range(0, text.length()).forEach(pos -> {
          final int word_id = text.at(pos);
          final Mx contextMat = contextMatrices.get(word_id);
          // Для каждого индекса
          range(0, dim).forEach(i -> {
            range(0, dim).forEach(j -> {
              final Mx dC = getContextMatDerivative(i, j);
              double diff = getContextDerivative(dC, txt, pos, window_left, window_right);
              contextMat.adjust(i, j, -step() * diff);
            });

            /*final Mx dSi = getContextSymMatDerivative(word_id, i);
            final Mx dKi = getContextSkewMatDerivative(word_id, i);
            double diffS = getContextDerivative(dSi, txt, pos, window_left, window_right);
            double diffK = getContextDerivative(dKi, txt, pos, window_left, window_right);

            contextSymVectors.adjust(word_id, i, -step() * diffS);
            contextSkewVectors.adjust(word_id, i, -step() * diffK);*/
          });
          contextMatrices.set(word_id, contextMat);
        });
      });
      for (int idx = 0; idx < vocab_size; idx++) {
        final Mx mat = contextMatrices.get(idx);
        VecTools.normalizeL2(mat);
        contextMatrices.set(idx, mat);
      }
      //project(contextSymVectors);
      //project(contextSkewVectors);
      log.info("Finished updating context vectors. Started image vectors updating. Time: " + Interval.time());

      // ОБНОВЛЯЕМ ВЕКТОР ОБРАЗА

      // Перебираем тексты
      range(0, texts_number).parallel().map(order::get).forEach(txt -> {
        final IntSeq text = text(txt);
        Mx C = VecTools.copy(C0);
        // Перебираем слова в тексте по очереди, чтобы их обновить вектора
        for (int pos = 0; pos < text.length(); pos++) {
          final int idx = text.at(pos);
          final Vec im = imageVectors.row(idx);
          final double derivativeTerm = getImageDerivativeTerm(im, C);

          // Для каждого индекса
          for (int i = 0; i < dim; i++) {
            final double derivative = getImageDerivative(im, C, derivativeTerm, i);
            imageVectors.adjust(idx, i, -step() * derivative);
          }
          final Mx context = getContextMat(idx);
          C = MxTools.multiply(C, context);
        }

      });
      project(imageVectors);
      //System.out.println("After project\n" + imageVectors.toString());
      log.info("Finished updating image vectors. Iter: " + iter + ". Time: " + Interval.time());
      range(0, texts_number).forEach(txt -> {

        log.info("Probability txt " + txt + " = " + logProbab(txt, 0, 0, 0));
      });

      //log.info("Iteration: " + iter + ", score: " + scoreCalculator.gloveScore() + ", time: " + Interval.time());
    }

    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < vocab_size; i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, imageVectors.row(i));
    }

    //log.info("Started checking derivative.");
    //checkDerivative(window_left, window_right);

    return new EmbeddingImpl<>(mapping);
  }

  private double initializeValue(int dim) {
    return (Math.random() - 0.5) / dim;
  }


  public Mx C0() {
    final Mx mat = new VecBasedMx(dim, dim);
    // Как там единичную матрицу задать функцией?
    VecTools.fill(mat, 0d);
    for (int i = 0; i < dim; i++) {
      mat.set(i, i, 1d);
    }
    return mat;
  }

  private void project(Mx mat) {
    for (int i = 0; i < mat.rows(); i++) {
      VecTools.normalizeL2(mat.row(i));
    }
  }

  private double getContextDerivative(Mx dContext_pos, int txt, int pos, int window_left, int window_right) {
    Mx C = VecTools.copy(C0);
    Mx dC = VecTools.copy(C0);
    final IntSeq text = text(txt);
    double diff = 0;

    //for (int t = Math.max(0, pos - window_left); t < Math.min(text.length(), pos + window_right + 1); t++) {
    for (int t = 0; t < text.length(); t++) {
      final int idx = text.at(t);
      final Vec im = imageVectors.row(idx);
      final Mx context = getContextMat(idx);
      if (t < pos) {
        C = MxTools.multiply(C, context);
      } else if (t == pos) {
        C = MxTools.multiply(C, context);
        dC = VecTools.copy(C);
      } else if (t == pos + 1) {
        dC = MxTools.multiply(dC, dContext_pos);
        C = MxTools.multiply(C, context);
        diff += getContextDerivativeTerm(im, C, dC);
      } else {
        dC = MxTools.multiply(dC, context);
        C = MxTools.multiply(C, context);
        diff += getContextDerivativeTerm(im, C, dC);
      }
    }
    return diff;
  }


  private double getContextDerivativeTerm(final Vec im, final Mx C, final Mx dC) {
    double result = VecTools.multiply(im, MxTools.multiply(dC, im));

    for (int i = 0; i < dict().size(); i++) {
      double softSum = 0d;
      final Vec u = imageVectors.row(i);
      double uCu = VecTools.multiply(u, MxTools.multiply(C, u));
      for (int j = 0; j < dict().size(); j++) {
        final Vec img = imageVectors.row(j);
        final double e = Math.exp(-VecTools.multiply(img, MxTools.multiply(C, img)) + uCu);
        /*if (e == POSITIVE_INFINITY) {
          uCu = MIN_PROBAB;
          softSum = 1d;
          break;
        }*/
        softSum += e;
      }
      result += -uCu / softSum;
    }

    return result;
  }

  private double getImageDerivative(final Vec im, final Mx C, final double derivativeTerm, int i) {
    double diff = 0d;
    for (int j = 0; j < im.dim(); j++) {
      diff += im.get(j) * (C.get(i, j) + C.get(j, i));
    }
    return diff * derivativeTerm;
  }

  private double getImageDerivativeTerm(final Vec im, final Mx C) {
    double softSum = 0d;
    final double uCu = VecTools.multiply(im, MxTools.multiply(C, im));
    for (int i = 0; i < dict().size(); i++) {
      final Vec img = imageVectors.row(i);
      final double e = Math.exp(-VecTools.multiply(img, MxTools.multiply(C, img)) + uCu);
      /*if (e == POSITIVE_INFINITY) {
        return 1d - MIN_PROBAB;
      }*/
      softSum += e;
    }
    return 1 - 1d / softSum;
  }

  public Mx getContextMat(int idx) {
    return contextMatrices.get(idx);
    /*final Vec s = contextSymVectors.row(idx);
    final Vec k = contextSkewVectors.row(idx);
    final Mx kkT = VecTools.outer(k, k);
    for (int i = 0; i < kkT.rows(); i++) {
      for (int j = 0; j < i; j++) {
        kkT.set(i, j, kkT.get(i, j) * -1d);
      }
    }
    return VecTools.append(VecTools.outer(s, s), kkT);*/
  }

  private Mx getContextMatDerivative(int di, int dj) {
    final Mx result = new VecBasedMx(dim, dim);
    VecTools.fill(result, 0d);
    result.set(di, dj, 1d);
    return result;
  }

  private Mx getContextSymMatDerivative(int idx, int di) {
    final Vec s = contextSymVectors.row(idx);
    final Mx result = new VecBasedMx(s.dim(), s.dim());
    VecTools.fill(result, 0d);
    for (int i = 0; i < s.dim(); i++) {
      result.set(di, i, s.get(i));
      result.set(i, di, s.get(i));
    }
    result.set(di, di, 2d * s.get(di));
    return result;
  }

  private Mx getContextSkewMatDerivative(int idx, int di) {
    final Vec k = contextSkewVectors.row(idx);
    final Mx result = new VecBasedMx(k.dim(), k.dim());
    VecTools.fill(result, 0d);
    for (int i = 0; i < k.dim(); i++) {
      final int sign = i > di ? -1 : 1;
      result.set(di, i, sign * k.get(i));
      result.set(i, di, -sign * k.get(i));
    }
    result.set(di, di, 2d * k.get(di));
    return result;
  }

  private void checkContextDerivative(final int window_left, final int window_right) {
    final double h = 0.001;
    final int txt = 0;
    final IntSeq text = text(txt);
    final int[] words = {0, 1, 2, text.length() / 2, text.length() - 1};
    final int[] indeces = {0, dim / 2, dim - 1};

    IntStream.of(words).forEach(pos -> {
      final int word_id = text.at(pos);
      log.info("Word " + pos + " sym vector " + contextSymVectors.row(word_id).toString());
      log.info("Word " + pos + " skew vector " + contextSkewVectors.row(word_id).toString());
      final double logProbab = logProbab(txt, pos, window_left, window_right);
      System.out.println(logProbab);

      // Для каждого индекса
      IntStream.of(indeces).forEach(i -> {
        final Mx dSi = getContextSymMatDerivative(word_id, i);
        final Mx dKi = getContextSkewMatDerivative(word_id, i);
        double diffS = getContextDerivative(dSi, txt, pos, window_left, window_right);
        double diffK = getContextDerivative(dKi, txt, pos, window_left, window_right);

        contextSymVectors.adjust(word_id, i, h);
        double realDiffS = (logProbab(txt, pos, window_left, window_right) - logProbab) / h;
        contextSymVectors.adjust(word_id, i, -h);
        contextSkewVectors.adjust(word_id, i, h);
        double realDiffK = (logProbab(txt, pos, window_left, window_right) - logProbab) / h;
        contextSkewVectors.adjust(word_id, i, -h);

        log.info("Index " + i + ": counted context sym = " + diffS);
        log.info("Index " + i + ": expected context sym = " + realDiffS);
        log.info("Index " + i + ": counted context skew = " + diffK);
        log.info("Index " + i + ": expected context skew = " + realDiffK);

      });

    });
  }

  private double logProbab(int txt, int pos, int window_left, int window_right) {
    final IntSeq text = text(txt);
    double probab = 1d;
    Mx C = VecTools.copy(C0);

    //for (int t = Math.max(0, pos - window_left); t < Math.min(text.length(), pos + window_right + 1); t++) {
    for (int t = 0; t < text.length(); t++) {
      final int idx = text.at(t);
      probab *= getProbability(C, idx);
      final Mx context = getContextMat(idx);
      C = MxTools.multiply(C, context);
    }
    return probab;
  }

  public double getProbability(Mx C, int image_idx) {
    final Vec im = imageVectors.row(image_idx);
    final double uCu = VecTools.multiply(im, MxTools.multiply(C, im));
    double softSum = 0d;
    for (int i = 0; i < dict().size(); i++) {
      final Vec img = imageVectors.row(i);
      final double e = Math.exp(-VecTools.multiply(img, MxTools.multiply(C, img)) + uCu);
      /*if (e == POSITIVE_INFINITY) {
        return MIN_PROBAB;
      }*/
      softSum += e;
    }
    return 1d / softSum;
  }

}

