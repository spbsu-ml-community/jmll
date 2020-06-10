package com.expleague.ml.embedding.lm;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.impl.LanguageModelBuiderBase;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntLongHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class LWMatrixMultBuilder extends LanguageModelBuiderBase {
  private static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  private static final double MIN_PROBAB = 1e-10;
  public static final double MAX_EIGEN_VALUE = 0.9;
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 10;
  private int dimDecomp = 0;
  private boolean isDecomposed = false;
  private FastRandom rng = new FastRandom();
  private boolean regularization = false;
  private LWMatrixRegression model;

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
    return this;
  }

  public LWMatrixMultBuilder isDecomposed(boolean flag) {
    this.isDecomposed = flag;
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
    final Vec theta = new ArrayVec(dim * (dim + 1) * wordsList.size() + dim * dim);
    { // parameter initialization
      VecTools.fillGaussian(theta, rng);
      final State initState = new State(theta, wordsList.size(), dim, isDecomposed);
      initState.initializeParams();
    }

    final Vec freqs = new ArrayVec(wordsList.size());
    final List<IntSeq> train = parsedTexts().subList(0, Math.min(5, parsedTexts().size()));
    train.stream().flatMapToInt(IntSeq::stream).forEach(idx -> freqs.adjust(idx, 1));
    final NCETarget target = new NCETarget(rng, freqs);
    for (int iter = 0; iter < T(); iter++, it++) {
      List<IntSeq> nextIter = new ArrayList<>(train);
      Collections.shuffle(nextIter);
      if (((it + 1) % 1) == 0) {
        final State state = fitSeq(train.get(0), theta, target, true);
      }
      nextIter.parallelStream().forEach(text -> {
        final Vec copy;
        synchronized (theta) {
          copy = VecTools.copy(theta);
        }
        final State state = fitSeq(text, copy, target, false);
        synchronized (theta) {
          state.commit(theta, step() / text.length());
        }
      });
    }

    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < wordsList.size(); i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, theta.sub(i * (dim * (dim + 1)) + dim * dim, dim));
    }

    return new EmbeddingImpl<>(mapping);
  }

  static long it = 0;
  private State fitSeq(IntSeq seq, Vec params, NCETarget target, boolean debug) {
    final State state = new State(params, dict().size(), dim, isDecomposed);
    DecimalFormat df = new DecimalFormat("#.####");
    df.setRoundingMode(RoundingMode.CEILING);
    double perplexity = 0;
    seq.stream().forEach(state::contextPrime);
    final List<Mx> parentContexts = new ArrayList<>();
    Mx currentContext = state.initialContext();
    Mx[] dUArr = new Mx[]{new VecBasedMx(dim, dim), new VecBasedMx(dim, dim)};
    if (debug && (it % 1000 == 0 || it == T() - 1))
      System.out.println("probs at iteration " + it);
    for (int t = 0; t < seq.length(); t++) {
      Mx dU = dUArr[t % 2];
      final int word = seq.intAt(t);
      if (debug) { // debug
        double wordLognom = MxTools.quadraticForm(currentContext, state.image(word));
        double denom = 0d;
        int counter = 0;
        for (int k = 0; k < wordsList.size(); k++) {
          if (target.p_n(k) > 0) {
            denom += Math.exp(MxTools.quadraticForm(currentContext, state.image(k)));
            counter ++;
          }
        }
        denom = Math.log(denom);
        if (it % 1000 == 0 || it == T() - 1) {
          double[] weights = new double[counter];
          int[] indexes = new int[counter];
          int sz = 0;
          for (int k = 0; k < wordsList.size(); k++) {
            if (target.p_n(k) > 0) {
              weights[sz] = -Math.exp(MxTools.quadraticForm(currentContext, state.image(k)) - denom);
              indexes[sz] = k;
              sz ++;
            }
          }
          ArrayTools.parallelSort(weights, indexes);
          StringBuilder stringBuilder = new StringBuilder();
          stringBuilder.append(wordsList.get(word)).append(":")
                  .append(df.format(Math.exp(wordLognom - denom))).append("\t->\t");
          for (int k  = 0; k < 5; k++) {
            stringBuilder.append(wordsList.get(indexes[k])).append(":")
                    .append(df.format(-weights[k])).append(", ");
          }
          System.out.println(stringBuilder.toString());
        }
        perplexity += -(wordLognom - denom) / seq.length();
      }

      if (Math.random() < 0.2) {
        target.step(currentContext, word, dU, state);
        double dUF = 1;
        // TODO: append step by initialContext
        for (int bp = t - 1; bp >= 0 && dUF > 1e-3; bp--) { // back propagation of the context error
        //for (int bp = t - 1; bp > t - 2 && dUF > 1e-3 && bp >= 0; bp--) {
          final Mx dUPrev = dUArr[bp % 2];
          final int bpWord = seq.intAt(bp);
          final Mx wordContext = state.context(bpWord);
          final Mx contextPrime = state.contextPrime(bpWord);
          final Mx parentContext = parentContexts.get(bp);
          dUF = 0.;
          for (int i = 0; i < dU.rows(); i++) {
            for (int j = 0; j < dU.columns(); j++) {
              final double dUPrev_ij = VecTools.multiply(dU.row(i), wordContext.row(j));
              dUPrev.set(i, j, dUPrev_ij);
              dUF += dUPrev_ij * dUPrev_ij;
              contextPrime.set(i, j, VecTools.multiply(dU.col(j), parentContext.col(i)));
            }
          }
          dUF = Math.sqrt(dUF) / dim;
          dU = dUPrev;
        }
      }
      final Mx context = state.context(word);
      parentContexts.add(currentContext);
      currentContext = MxTools.multiply(currentContext, context);
      //currentContext = context;
    }
    if (debug && (it % 1000 == 0 || it == T() - 1)) {
      System.out.println("Perplexity: " + Math.exp(perplexity));
      System.out.println();
    }
    return state;
  }

  private static class State {
    private final Mx[] context;
    private final Vec[] image;

    private final Vec parametersOrig;
    private final int dim, dictSize;
    private final boolean isDecomposed;

    State(Vec parametersOrig, int dictSize, int dim, boolean isDecomposed) {
      this.parametersOrig = parametersOrig;
      this.dictSize = dictSize;
      this.context = new Mx[dictSize];
      this.image = new Vec[dictSize];
      //noinspection IntegerDivisionInFloatingPointContext
      this.dim = dim;
      this.isDecomposed = isDecomposed;
    }

    public void initializeParams() {
      Mx L = new VecBasedMx(dim, dim);
      Mx Q = new VecBasedMx(dim, dim);
      for (int i = 0; i < dictSize; i++) {
        int start = i * (dim * (dim + 1)) + dim * dim;
        //normalizeVec(parametersOrig.sub(start, dim));
        start = i * (dim * (dim + 1));
        final Mx context = new VecBasedMx(dim, parametersOrig.sub(start, dim * dim));
        //IntStream.range(0, dim).forEach(j -> normalizeVec(context.row(j)));
        IntStream.range(0, dim).forEach(j -> normalizeMx(context, L, Q));
//        VecTools.scale(context, MAX_EIGEN_VALUE / (MxTools.mainEigenValue(context) + 1e-6));
      }
      Mx initContext = new VecBasedMx(dim, parametersOrig.sub(parametersOrig.length() - dim * dim, dim * dim));
      VecTools.assign(initContext, MxTools.E(dim));
    }

    private void normalizeVec(Vec vec) {
      VecTools.normalizeL2(vec);
    }

    private void normalizeMx(Mx mx, Mx L, Mx Q) {
      MxTools.householderLQ(mx, L, Q);
      VecTools.assign(mx, Q);
    }

    public void commit(Vec parametersOrig, double step) {
      Mx L = new VecBasedMx(dim, dim);
      Mx Q = new VecBasedMx(dim, dim);
      for (int i = 0; i < this.context.length; i++) {
        if (this.context[i] == null || this.image[i] == null)
          continue;
        {
          final int start = i * (dim * (dim + 1));
          final Mx context = new VecBasedMx(dim, parametersOrig.sub(start, dim * dim));
          VecTools.incscale(context, this.context[i], step);
          //IntStream.range(0, dim).forEach(j -> normalizeVec(context.row(j)));
          IntStream.range(0, dim).forEach(j -> normalizeMx(context, L, Q));
//          final double mev = Math.abs(MxTools.mainEigenValue(context));//, VecTools.maxMod(context));
//          VecTools.scale(context, MAX_EIGEN_VALUE / (mev + 1e-6));
        }
        {
          final int start = i * (dim * (dim + 1)) + dim * dim;
          final Vec image = parametersOrig.sub(start, dim);
          if (isDecomposed && context[i] != null) {
            final Vec imageDer = VecTools.append(MxTools.multiply(this.context[i], image), MxTools.multiply(MxTools.transpose(this.context[i]), image));
            VecTools.incscale(image, imageDer, step);
          } else {
            VecTools.incscale(image, this.image[i], step);
          }
          //normalizeVec(image);
        }
      }
    }

    public Mx context(int i) {
      final int start = i * (dim * (dim + 1));
      if (isDecomposed) {
        final Mx sym = VecTools.outer(image(i), image(i));
        return VecTools.append(sym, parametersOrig.sub(start, dim * dim));
      } else {
        return new VecBasedMx(dim, parametersOrig.sub(start, dim * dim));
      }
    }

    public Vec image(int i) {
      final int start = i * (dim * (dim + 1)) + dim * dim;
      return parametersOrig.sub(start, dim);
    }

    public Vec imagePrime(int i) {
      return image[i] != null ? image[i] : (image[i] = new ArrayVec(dim));
    }

    public Mx contextPrime(int i) {
      return context[i] != null ? context[i] : (context[i] = new VecBasedMx(dim, dim));
    }

    public Mx initialContext() {
      return new VecBasedMx(dim, parametersOrig.sub(parametersOrig.length() - dim * dim, dim * dim));
    }
  }

  public class NCETarget {
    private final FastRandom rng;
    private final Vec freqs;

    public NCETarget(FastRandom rng, Vec freqs) {
      this.rng = rng;
      this.freqs = freqs;
      VecTools.normalizeL1(freqs);
    }

    public double p_n(int wordIdx) {
      return freqs.get(wordIdx);
    }

    public void step(Mx context, int xidx, Mx contextDer, State state) {
      //int yidx;
      int y_num = 1;
      TIntList yidxs = new TIntArrayList(y_num);
      //noinspection StatementWithEmptyBody
      //while ((yidx = nextP_n()) == xidx);
      for (int k = 0; k < y_num; k++) {
        int yidx = xidx;
        boolean flag = true;
        while (flag) {
          flag = false;
          while ((yidx = nextP_n()) == xidx) ;
          while (yidxs.contains(yidx)) {
            yidx = nextP_n();
            flag = true;
          }
        }
        yidxs.add(yidx);

      }

      final double G_x = G(context, state, xidx);
      //final double G_y = G(context, state, yidx);
      final double[] G_ys = new double[y_num];
      for (int k = 0; k < y_num; k++) {
        G_ys[k] = G(context, state, yidxs.get(k));
      }

      final double sigmaG_x = 1. / (1. + Math.exp(-G_x));
      //final double sigmaG_y = 1. / (1. + Math.exp(-G_y));
      final double[] sigmaG_ys = new double[y_num];
      for (int k = 0; k < y_num; k++) {
        sigmaG_ys[k] = 1. / (1. + Math.exp(-G_ys[k]));
      }

      final Vec ximage = state.image(xidx);
      //final Vec yimage = state.image(yidx);
      final Vec[] yimages = new Vec[y_num];
      for (int k = 0; k < y_num; k++) {
        yimages[k] = state.image(yidxs.get(k));
      }


      { // x, y image derivative
        final Vec ximagePrime = state.imagePrime(xidx);
        //final Vec yimagePrime = state.imagePrime(yidx);
        final Vec[] yimagePrimes = new Vec[y_num];
        for (int i = 0; i < y_num; i++) {
          yimagePrimes[i] = state.imagePrime(yidxs.get(i));
        }

        for (int i = 0; i < ximage.dim(); i++) {
          for (int j = 0; j < ximage.dim(); j++) {
            final double contextW = context.get(i, j) + context.get(j, i);
            ximagePrime.adjust(i, (1 - sigmaG_x) * ximage.get(j) * contextW);
            //yimagePrime.adjust(i, -sigmaG_y * yimage.get(j) * contextW);
            for (int k = 0; k < y_num; k++) {
              yimagePrimes[k].adjust(i, -sigmaG_ys[k] * yimages[k].get(j) * contextW);
            }
          }
        }
      }
      { // contextDer
        VecTools.scale(contextDer, 0.);
        VecTools.incscale(contextDer, VecTools.outer(ximage, ximage), 1 - sigmaG_x);
        //VecTools.incscale(contextDer, VecTools.outer(yimage, yimage), -sigmaG_y);
        for (int k = 0; k < y_num; k++) {
          VecTools.incscale(contextDer, VecTools.outer(yimages[k], yimages[k]), -sigmaG_ys[k]);
        }
      }
    }

    private double G(Mx context, State state, int idx) {
      return MxTools.quadraticForm(context, state.image(idx)) - Math.log(p_n(idx));
    }

    public int nextP_n() {
      return rng.nextSimple(freqs, 1.);
    }
  }
}

