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
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.impl.LanguageModelBuiderBase;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntLongHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.IntStream;

public class LWMatrixMultBuilder extends LanguageModelBuiderBase {
  private static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  private static final double MIN_PROBAB = 1e-10;
  public static final double MAX_EIGEN_VALUE = 0.9;
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 10;
  private int dimDecomp = 0;
  private FastRandom rng = new FastRandom();
  private boolean regularization = false;

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

  public LWMatrixMultBuilder dimDecomp(int dim) {
    this.dimDecomp = dim;
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
      final State initState = new State(theta, wordsList.size());
      for (int i = 0; i < wordsList.size(); i++) {
        VecTools.normalizeL2(initState.image(i));
        final Mx context = initState.context(i);
//        VecTools.assign(context, MxTools.E(dim));
        IntStream.range(0, dim).forEach(j -> VecTools.normalizeL2(context.row(j)));
//        VecTools.scale(context, MAX_EIGEN_VALUE / (MxTools.mainEigenValue(context) + 1e-6));
      }
      VecTools.assign(initState.initialContext(), MxTools.E(dim));
    }

    final Vec freqs = new ArrayVec(wordsList.size());
    final List<IntSeq> train = parsedTexts().subList(0, 100);
    train.stream().flatMapToInt(IntSeq::stream).forEach(idx -> freqs.adjust(idx, 1));
    final NCETarget target = new NCETarget(rng, freqs);
    for (int iter = 0; iter < T(); iter++, it++) {
      List<IntSeq> nextIter = new ArrayList<>(train);
      Collections.shuffle(nextIter);
      if (((it + 1) % 1) == 0) {
        final State state = fitSeq(train.get(0), theta, target, true);
//        for (int i = 0; i < wordsList.size(); i++){
//          if (state.image[i] != null) {
//            System.out.println(state.context[i]);
//            System.out.println(state.image[i]);
//          }
//        }
      }
      nextIter.parallelStream().forEach(text -> {
        final Vec copy;
        synchronized (theta) {
          copy = VecTools.copy(theta);
        }
        final State state = fitSeq(text, copy, target, false);
        synchronized (theta) {
          state.commit(theta, step());
        }
      });
    }

//    final Mx imageVectors = model.getImageVectors();
//    final Map<CharSeq, Vec> mapping = new HashMap<>();
//    for (int i = 0; i < wordsList.size(); i++) {
//      final CharSeq word = dict().get(i);
//      mapping.put(word, imageVectors.row(i));
//      System.out.println(imageVectors.row(i));
//    }
//
    return null;//new EmbeddingImpl<>(mapping);
  }

  static long it = 0;
  private State fitSeq(IntSeq seq, Vec params, NCETarget target, boolean debug) {
    final State state = new State(params, dict().size());
    double perplexity = 0;
    seq.stream().forEach(state::contextPrime);
    final List<Mx> parentContexts = new ArrayList<>();
    Mx currentContext = state.initialContext();
    Mx[] dUArr = new Mx[]{new VecBasedMx(dim, dim), new VecBasedMx(dim, dim)};
    if (debug)
      System.out.println("probs at iteration " + it);
    for (int t = 0; t < seq.length(); t++) {
      Mx dU = dUArr[t % 2];
      final int word = seq.intAt(t);
      if (debug) { // debug
//        System.out.print(wordsList.get(word) + "\t");
          final double logNom = MxTools.quadraticForm(currentContext, state.image(word));
          double denom = 0;
          for (int k = 0; k < wordsList.size(); k++) {
            if (target.p_n(k) > 0)
              denom += Math.exp(MxTools.quadraticForm(currentContext, state.image(k)));
          }
//          System.out.println(Math.exp(logNom - Math.log(denom)));
          perplexity += -(logNom - Math.log(denom)) / seq.length();
      }
      target.step(currentContext, word, dU, state);
      double dUF = 1;
      // TODO: append step by initialContext
      for (int bp = t - 1; bp >= 0 && dUF > 1e-3; bp--) { // back propagation of the context error
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
      final Mx context = state.context(word);
      parentContexts.add(currentContext);
      currentContext = MxTools.multiply(currentContext, context);
    }
    if (debug) {
      System.out.println("Perplexity: " + Math.exp(perplexity));
      System.out.println();
    }
    return state;
  }

  private static class State {
    private final Mx[] context;
    private final Vec[] image;

    private final Vec parametersOrig;
    private final int dim;

    State(Vec parametersOrig, int dictSize) {
      this.parametersOrig = parametersOrig;
      this.context = new Mx[dictSize];
      this.image = new Vec[dictSize];
      //noinspection IntegerDivisionInFloatingPointContext
      this.dim = (int)Math.ceil((-1 + Math.sqrt(1 + 4 * (parametersOrig.dim() / dictSize))) / 2);
    }

    public void commit(Vec parametersOrig, double step) {
      for (int i = 0; i < context.length; i++) {
        if (context[i] == null)
          continue;
        {
          final int start = i * (dim * (dim + 1));
          final Mx context = new VecBasedMx(dim, parametersOrig.sub(start, dim * dim));
          VecTools.incscale(context, this.context[i], step);
          IntStream.range(0, dim).forEach(j -> VecTools.normalizeL2(context.row(j)));
//          final double mev = Math.abs(MxTools.mainEigenValue(context));//, VecTools.maxMod(context));
//          VecTools.scale(context, MAX_EIGEN_VALUE / (mev + 1e-6));
        }
        {
          final int start = i * (dim * (dim + 1)) + dim * dim;
          final Vec image = parametersOrig.sub(start, dim);
          VecTools.incscale(image, this.image[i], step);
          VecTools.normalizeL2(image);
        }
      }
    }

    public Mx context(int i) {
      final int start = i * (dim * (dim + 1));
      return new VecBasedMx(dim, parametersOrig.sub(start, dim * dim));
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
      int yidx;
      //noinspection StatementWithEmptyBody
      while ((yidx = nextP_n()) == xidx);
      final double G_x = G(context, state, xidx);
      final double G_y = G(context, state, yidx);
      final double sigmaG_x = 1. / (1. + Math.exp(-G_x));
      final double sigmaG_y = 1. / (1. + Math.exp(-G_y));

      final Vec ximage = state.image(xidx);
      final Vec yimage = state.image(yidx);

      { // x, y image derivative
        final Vec ximagePrime = state.imagePrime(xidx);
        final Vec yimagePrime = state.imagePrime(yidx);
        for (int i = 0; i < ximage.dim(); i++) {
          for (int j = 0; j < ximage.dim(); j++) {
            final double contextW = context.get(i, j) + context.get(j, i);
//            if (contextW > 10)
//              System.out.println();
            ximagePrime.adjust(i, (1 - sigmaG_x) * ximage.get(j) * contextW);
            yimagePrime.adjust(i, -sigmaG_y * yimage.get(j) * contextW);
          }
        }
      }
      { // contextDer
        VecTools.scale(contextDer, 0.);
        VecTools.incscale(contextDer, VecTools.outer(ximage, ximage), 1 - sigmaG_x);
        VecTools.incscale(contextDer, VecTools.outer(yimage, yimage), -sigmaG_y);
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

