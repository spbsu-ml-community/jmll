package com.expleague.ml.embedding.LM;

import com.expleague.commons.math.FuncC1;
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
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.optimization.Optimize;
import com.expleague.ml.optimization.impl.AdamDescent;
import com.expleague.ml.optimization.impl.GradientDescent;
import com.expleague.ml.optimization.impl.SAGADescent;
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
  public LWMatrixRegression model;

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
    rng = new FastRandom();

    contextSymVectors = new VecBasedMx(vocab_size, dim);
    contextSkewVectors = new VecBasedMx(vocab_size, dim);
    List<Mx> contextSymMats = new ArrayList<>(vocab_size);
    List<Mx> contextSkewMats = new ArrayList<>(vocab_size);
    contextMatrices = new ArrayList<>(vocab_size);
    imageVectors = new VecBasedMx(vocab_size, dim);

    for (int i = 0; i < vocab_size; i++) {
      Mx mat = new VecBasedMx(dim, dim);
      Mx matS = new VecBasedMx(2, dim);
      Mx matK = new VecBasedMx(2, dim);
      for (int j = 0; j < dim; j++) {
        contextSymVectors.set(i, j, initializeValue(dim));
        contextSkewVectors.set(i, j, initializeValue(dim));
        imageVectors.set(i, j, initializeValue(dim));
        matS.set(0, j, initializeValue(dim));
        matS.set(1, j, initializeValue(dim));
        matK.set(0, j, initializeValue(dim));
        matK.set(1, j, initializeValue(dim));
      }
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < j; k++) {
          mat.set(j, k, contextSymVectors.get(i, j) * contextSymVectors.get(i, k) - contextSkewVectors.get(i, j) * contextSkewVectors.get(i, k));
          mat.set(j, k, initializeValue(dim));
        }
        for (int k = j; k < dim; k++) {
          mat.set(j, k, contextSymVectors.get(i, j) * contextSymVectors.get(i, k) + contextSkewVectors.get(i, j) * contextSkewVectors.get(i, k));
          mat.set(j, k, initializeValue(dim));
        }
      }
      //VecTools.normalizeL2(mat);
      contextMatrices.add(mat);
      VecTools.normalizeL2(matS);
      contextSymMats.add(matS);
      VecTools.normalizeL2(matK);
      contextSkewMats.add(matK);
      //VecTools.normalizeL2(contextSymVectors.row(i));
      //VecTools.normalizeL2(contextSkewVectors.row(i));
      //VecTools.normalizeL2(imageVectors.row(i));
    }

    /*for (int i = 0; i < vocab_size; i++) {
      System.out.println(dict().get(i) + "  " + imageVectors.row(i).toString());
    }*/

    System.out.println("MATRIX");

    final LWSimpleRegression[] imageFuncsList = new LWSimpleRegression[1];
    imageFuncsList[0] = new LWSimpleRegression(vocab_size, dim, text(0));
    final Vec funcWeights = new ArrayVec(1);
    funcWeights.set(0, 1d);
    final FuncEnsemble<FuncC1> imageFuncs = new FuncEnsemble<>(imageFuncsList, funcWeights);
    final RegularizerFunc imageReg = new LWSimpleRegression.LWRegularizer(vocab_size, dim);

    //final SAGADescent imageOptimizer = new SAGADescent(step(), T(), rng, System.out);
    final GradientDescent imageOptimizer = new GradientDescent(LWSimpleRegression.fold(contextMatrices, imageVectors), 1e-4);
    //final AdamDescent imageOptimizer = new AdamDescent(rng, T(), 1, step());

    Vec result = imageOptimizer.optimize(imageFuncsList[0], imageReg, LWSimpleRegression.fold(contextMatrices, imageVectors));
    /*imageVectors = LWSimpleRegression.unfoldImages(result, vocab_size, dim);
    contextMatrices = LWSimpleRegression.unfoldContexts(result, vocab_size, dim);
    for (int i = 0; i < vocab_size; i++) {
      System.out.println(dict().get(i) + "  " + imageVectors.row(i).toString());
    }*/
    model = imageFuncsList[0];

    System.out.println("SKS");

    /*final LWSksRegresseion[] imageFuncsList = new LWSksRegresseion[1];
    imageFuncsList[0] = new LWSksRegresseion(vocab_size, dim, text(0));
    final Vec funcWeights = new ArrayVec(1);
    funcWeights.set(0, 1d);
    final FuncEnsemble<FuncC1> imageFuncs = new FuncEnsemble<>(imageFuncsList, funcWeights);
    final RegularizerFunc imageReg = new LWSksRegresseion.LWRegularizer(vocab_size, dim);

    //final SAGADescent imageOptimizer = new SAGADescent(step(), T(), rng, System.out);
    final GradientDescent imageOptimizer = new GradientDescent(LWSksRegresseion.fold(contextSymVectors, contextSkewVectors, imageVectors), 1e-3);
    //final AdamDescent imageOptimizer = new AdamDescent(rng, T(), 1, step());

    Vec result = imageOptimizer.optimize(imageFuncsList[0], imageReg, LWSksRegresseion.fold(contextSymVectors, contextSkewVectors, imageVectors));
    /*imageVectors = LWSksRegresseion.unfoldImages(result, vocab_size, dim);
    contextSymVectors = LWSksRegresseion.unfoldSymmetricContexts(result, vocab_size, dim);
    contextSkewVectors = LWSksRegresseion.unfoldSkewsymmetricContexts(result, vocab_size, dim);

    for (int i = 0; i < vocab_size; i++) {
      System.out.println(dict().get(i) + "  " + imageVectors.row(i).toString());
    }
    System.out.println(contextSymVectors.toString());*/
    //model = imageFuncsList[0];

    System.out.println("BOTH");

    /*final BothRegression imageFuncsList = new BothRegression(vocab_size, dim, text(0));
    final RegularizerFunc imageReg = new BothRegression.LWRegularizer(vocab_size, dim);

    final GradientDescent imageOptimizer = new GradientDescent(BothRegression.fold(contextMatrices, contextSymVectors, contextSkewVectors, imageVectors), 1e-3);

    Vec result = imageOptimizer.optimize(imageFuncsList, imageReg, BothRegression.fold(contextMatrices, contextSymVectors, contextSkewVectors, imageVectors));

    */System.out.println("TMP");

    /*final TmpRegresseion[] imageFuncsList = new TmpRegresseion[1];
    imageFuncsList[0] = new TmpRegresseion(vocab_size, dim, text(0));
    final Vec funcWeights = new ArrayVec(1);
    funcWeights.set(0, 1d);
    final FuncEnsemble<FuncC1> imageFuncs = new FuncEnsemble<>(imageFuncsList, funcWeights);
    final RegularizerFunc imageReg = new TmpRegresseion.LWRegularizer(vocab_size, dim);

    //final SAGADescent imageOptimizer = new SAGADescent(step(), T(), rng, System.out);
    final GradientDescent imageOptimizer = new GradientDescent(TmpRegresseion.fold(contextSymMats, contextSkewMats, imageVectors), 1e-3);
    //final AdamDescent imageOptimizer = new AdamDescent(rng, T(), 1, step());

    imageOptimizer.optimize(imageFuncsList[0], imageReg, TmpRegresseion.fold(contextSymMats, contextSkewMats, imageVectors));
    model = imageFuncsList[0];*/

    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < vocab_size; i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, imageVectors.row(i));
    }

    return new EmbeddingImpl<>(mapping);
  }

  private double initializeValue(int dim) {
    return (Math.random() - 0.5) / dim;
  }

}

