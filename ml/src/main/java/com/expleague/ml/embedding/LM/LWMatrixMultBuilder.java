package com.expleague.ml.embedding.LM;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeq;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import com.expleague.ml.embedding.impl.EmbeddingImpl;
import com.expleague.ml.embedding.impl.LanguageModelBuiderBase;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.optimization.impl.GradientDescent;
import gnu.trove.map.TObjectIntMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LWMatrixMultBuilder extends LanguageModelBuiderBase {
  private static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  private static final double MIN_PROBAB = 1e-10;
  private double xMax = 10;
  private double alpha = 0.75;
  private int dim = 10;
  private int dimDecomp = 0;
  private FastRandom rng = new FastRandom();
  private boolean regularization = false;
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
    final int vocab_size = dict().size();
    final int texts_number = textsNumber();
    final int windowLeft = wleft();
    final int windowRight = wright();
    rng = new FastRandom();

    if (dimDecomp == 0) {
      System.out.println("LW Matrix");
      model = new LWSimpleRegression(text(0), vocab_size, dim, windowLeft, windowRight);
      final RegularizerFunc imageReg = new LWSimpleRegression.LWRegularizer(vocab_size, dim);

      //final SAGADescent imageOptimizer = new SAGADescent(step(), T(), rng, System.out);
      final GradientDescent imageOptimizer = new GradientDescent(model.getParameters(), 1e-3);
      //final AdamDescent imageOptimizer = new AdamDescent(rng, T(), 1, step());

      imageOptimizer.optimize(model, imageReg, model.getParameters());
    } else if (dimDecomp == -1) {
      System.out.println("LW Both Debug");
      /*final BothRegression imageFuncsList = new BothRegression(vocab_size, dim, text(0));
      final RegularizerFunc imageReg = new BothRegression.LWRegularizer(vocab_size, dim);

      final GradientDescent imageOptimizer = new GradientDescent(BothRegression.fold(contextMatrices, contextSymVectors, contextSkewVectors, imageVectors), 1e-3);

      Vec result = imageOptimizer.optimize(imageFuncsList, imageReg, BothRegression.fold(contextMatrices, contextSymVectors, contextSkewVectors, imageVectors));
      model = imageFunc;*/
    } else {
      System.out.println("LW Sks Decomposition with rank " + dimDecomp);

      model = new LWSksRegression(text(0), vocab_size, dim, dimDecomp);
      final RegularizerFunc imageReg = new LWSksRegression.LWRegularizer(vocab_size, dim, dimDecomp);

      //final SAGADescent imageOptimizer = new SAGADescent(step(), T(), rng, System.out);
      final GradientDescent imageOptimizer = new GradientDescent(model.getParameters(), 1e-3);
      //final AdamDescent imageOptimizer = new AdamDescent(rng, T(), 1, step());

      imageOptimizer.optimize(model, imageReg, model.getParameters());
    }

    final Mx imageVectors = model.getImageVectors();
    final Map<CharSeq, Vec> mapping = new HashMap<>();
    for (int i = 0; i < vocab_size; i++) {
      final CharSeq word = dict().get(i);
      mapping.put(word, imageVectors.row(i));
      System.out.println(imageVectors.row(i));
    }

    return new EmbeddingImpl<>(mapping);
  }

}

