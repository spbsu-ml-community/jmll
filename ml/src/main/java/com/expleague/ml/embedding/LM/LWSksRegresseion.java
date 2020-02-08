package com.expleague.ml.embedding.LM;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.func.RegularizerFunc;

import java.util.ArrayList;
import java.util.List;

import static java.util.stream.IntStream.range;

public class LWSksRegresseion  extends LWMatrixRegression {
  private Mx contextSymVectors, contextSkewVectors;

  public LWSksRegresseion(int dimx, int dimy, IntSeq text) {
    super(dimx, dimy, text);
    this.dimTotal = dimX * dimY * 2 + dimX * dimY;

    contextSymVectors = new VecBasedMx(dimX, dimY);
    contextSkewVectors = new VecBasedMx(dimX, dimY);

    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        contextSymVectors.set(i, j, initializeValue(dimY));
        contextSkewVectors.set(i, j, initializeValue(dimY));
      }
      //VecTools.normalizeL2(contextSymVectors.row(i));
      //VecTools.normalizeL2(contextSkewVectors.row(i));
    }
  }

  @Override
  public Vec L(Vec at) {
    return VecTools.fill(super.L(at), 100);
  }

  @Override
  public Mx getContextMat(int idx) {
    final Vec s = contextSymVectors.row(idx);
    final Vec k = contextSkewVectors.row(idx);
    final Mx kkT = VecTools.outer(k, k);
    for (int i = 0; i < kkT.rows(); i++) {
      for (int j = 0; j < i; j++) {
        kkT.set(i, j, kkT.get(i, j) * -1d);
      }
    }
    return VecTools.append(VecTools.outer(s, s), kkT);
  }

  // matconcat: dimX строк, в каждой dimY симметричных контекстных + dimY асимметричных контекстных + dimY образ
  @Override
  public double value(Vec in) {
    unfold(in);
    return value();
  }

  @Override
  public int dim() {
    return dimTotal;
  }

  @Override
  public Vec gradient(Vec in) {
    unfold(in);
    Mx grads = new VecBasedMx(dimX, dimY * 3);

    // Перебираем слова в тексте по очереди, чтобы их обновить вектора
    range(0, text.length()).forEach(pos -> {
      final int word_id = text.at(pos);
      // Для каждого индекса
      range(0, dimY).forEach(i -> {
        final Mx dSi = getContextSymMatDerivative(word_id, i);
        final Mx dKi = getContextSkewMatDerivative(word_id, i);
        double diffS = getContextDerivative(dSi, pos);
        double diffK = getContextDerivative(dKi, pos);

        grads.adjust(word_id, i, diffS);
        grads.adjust(word_id, i + dimY, diffK);
      });
    });

    Mx C = VecTools.copy(C0);
    for (int pos = 0; pos < text.length(); pos++) {
      final int idx = text.at(pos);
      final Vec im = imageVectors.row(idx);
      final double derivativeTerm = getImageDerivativeTerm(im, C);

      // Для каждого индекса
      for (int j = 0; j < dimY; j++) {
        grads.adjust(idx, j + dimY * 2, getImageDerivative(im, C, derivativeTerm, j));
      }
      C = MxTools.multiply(C, getContextMat(idx));
    }
    double sqim1 = VecTools.multiply(imageVectors.row(0), imageVectors.row(0));
    double sqim2 = VecTools.multiply(imageVectors.row(1), imageVectors.row(1));
    Mx c1 = getContextMat(0);
    double im1C1im1 = VecTools.multiply(imageVectors.row(0), MxTools.multiply(c1, imageVectors.row(0)));
    double im2C1im2 = VecTools.multiply(imageVectors.row(1), MxTools.multiply(c1, imageVectors.row(1)));
    double dim00 = 2. * imageVectors.get(0, 0)  / (1. + Math.exp(sqim2 - sqim1));
    double dim01 = 2. * imageVectors.get(0, 1)  / (1. + Math.exp(sqim2 - sqim1));
    double dim10 = (2. * imageVectors.get(1, 0) * c1.get(0, 0) +
        imageVectors.get(1, 1) * (c1.get(0,1) + c1.get(1, 0))) / (1. + Math.exp(im1C1im1 - im2C1im2));
    double dim11 = (2. * imageVectors.get(1, 1) * c1.get(1, 1) +
        imageVectors.get(1, 0) * (c1.get(0,1) + c1.get(1, 0))) / (1. + Math.exp(im1C1im1 - im2C1im2));
    /*double im1s10im1 = 2. * imageVectors.get(0, 0) * (imageVectors.get(0, 0) * contextSymVectors.get(0, 0) +
        imageVectors.get(0, 1) * contextSymVectors.get(0, 1));
    double im2s10im2 = 2. * imageVectors.get(1, 0) * (imageVectors.get(1, 0) * contextSymVectors.get(0, 0) +
        imageVectors.get(1, 1) * contextSymVectors.get(0, 1));
    double im1s11im1 = 2. * imageVectors.get(0, 1) * (imageVectors.get(0, 0) * contextSymVectors.get(0, 0) +
        imageVectors.get(0, 1) * contextSymVectors.get(0, 1));
    double im2s11im2 = 2. * imageVectors.get(1, 1) * (imageVectors.get(1, 0) * contextSymVectors.get(0, 0) +
        imageVectors.get(1, 1) * contextSymVectors.get(0, 1));
    double im1k10im1 = 2. * contextSkewVectors.get(0, 0) * imageVectors.get(0, 0) * imageVectors.get(0, 0);
    double im2k10im2 = 2. * contextSkewVectors.get(0, 0) * imageVectors.get(1, 0) * imageVectors.get(1, 0);;
    double im1k11im1 = 2. * contextSkewVectors.get(0, 1) * imageVectors.get(0, 1) * imageVectors.get(0, 1);
    double im2k11im2 = 2. * contextSkewVectors.get(0, 1) * imageVectors.get(1, 1) * imageVectors.get(1, 1);
    double ds10 = im2s10im2 + (Math.exp(-im1C1im1) * -im1s10im1 + Math.exp(-im2C1im2) * -im2s10im2) / (Math.exp(-im1C1im1) + Math.exp(-im2C1im2));
    double ds11 = im2s11im2 + (Math.exp(-im1C1im1) * -im1s11im1 + Math.exp(-im2C1im2) * -im2s11im2) / (Math.exp(-im1C1im1) + Math.exp(-im2C1im2));
    double dk10 = im2k10im2 + (Math.exp(-im1C1im1) * -im1k10im1 + Math.exp(-im2C1im2) * -im2k10im2) / (Math.exp(-im1C1im1) + Math.exp(-im2C1im2));
    double dk11 = im2k11im2 + (Math.exp(-im1C1im1) * -im1k11im1 + Math.exp(-im2C1im2) * -im2k11im2) / (Math.exp(-im1C1im1) + Math.exp(-im2C1im2));*/
    return grads;
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
      final int sign = i >= di ? 1 : -1;
      result.set(di, i, sign * k.get(i));
      result.set(i, di, -sign * k.get(i));
    }
    result.set(di, di, 2d * k.get(di));
    return result;
  }

  private void unfold(Vec in) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 3, in);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        contextSymVectors.set(i, j, matrices.get(i, j));
        contextSkewVectors.set(i, j, matrices.get(i, j + dimY));
        imageVectors.set(i, j, matrices.get(i, j + 2 * dimY));
      }
    }
  }

  public static Mx unfoldImages(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 3, in);
    Mx result = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        result.set(i, j, matrices.get(i, j + 2 * dimY));
      }
    }
    return result;
  }

  public static Mx unfoldSymmetricContexts(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 3, in);
    Mx result = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        result.set(i, j, matrices.get(i, j ));
      }
    }
    return result;
  }

  public static Mx unfoldSkewsymmetricContexts(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 3, in);
    Mx result = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        result.set(i, j, matrices.get(i, j + dimY));
      }
    }
    return result;
  }

  public static Mx fold(Mx contextSym, Mx contextSkew, Mx images) {
    int dimX = images.rows();
    int dimY = images.columns();
    Mx res = new VecBasedMx(dimX, dimY * 3);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        res.set(i, j, contextSym.get(i, j));
        res.set(i, j + dimY, contextSkew.get(i, j));
        res.set(i, j + dimY * 2, images.get(i, j));
      }
    }
    return res;
  }

  public static class LWRegularizer extends RegularizerFunc.Stub {

    private final int dimX, dimY;

    public LWRegularizer(int dimX, int dimY) {
      this.dimX = dimX;
      this.dimY = dimY;
    }

    @Override
    public Vec project(Vec in) {
      /*Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 3, in);
      for (int i = 0; i < dimX; i++) {
        Vec sym, skew, im;
        sym = matrices.row(i).sub(0, dimY);
        skew = matrices.row(i).sub(dimY, dimY);
        im = matrices.row(i).sub(2 * dimY, dimY);
        VecTools.normalizeL2(sym);
        VecTools.normalizeL2(skew);
        VecTools.normalizeL2(im);
        for (int j = 0; j < dimY; j++) {
          matrices.set(i, j, sym.get(j));
          matrices.set(i, j + dimY, skew.get(j));
          matrices.set(i, j + dimY * 2, im.get(j));
        }
      }
      return matrices;*/
      return in;
    }

    // SAGADescent вроде его не юзает
    @Override
    public double value(Vec x) {
      return 0d;
    }

    @Override
    public int dim() {
      return dimX * dimY * 3;
    }
  }
}
