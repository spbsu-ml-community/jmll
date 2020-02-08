package com.expleague.ml.embedding.LM;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.func.RegularizerFunc;

import java.util.ArrayList;
import java.util.List;

import static java.util.stream.IntStream.range;

public class TmpRegresseion extends LWMatrixRegression {
  private List<Mx> contextSymVectors, contextSkewVectors;

  public TmpRegresseion(int dimx, int dimy, IntSeq text) {
    super(dimx, dimy, text);
    this.dimTotal = dimX * 5 * dimY;

    contextSymVectors = new ArrayList<>(dimX);
    contextSkewVectors = new ArrayList<>(dimX);
    imageVectors = new VecBasedMx(dimX, dimY);

    for (int i = 0; i < dimX; i++) {
      Mx matS = new VecBasedMx(2, dimY);
      Mx matK = new VecBasedMx(2, dimY);
      for (int j = 0; j < 2 * dimY; j++) {
        matS.set(0, j % dimY, initializeValue(dimY));
        matS.set(1, j % dimY, initializeValue(dimY));
        matK.set(0, j % dimY, initializeValue(dimY));
        matK.set(1, j % dimY, initializeValue(dimY));
      }
      //VecTools.normalizeL2(contextSymVectors.row(i));
      //VecTools.normalizeL2(contextSkewVectors.row(i));
      contextSymVectors.add(matS);
      contextSkewVectors.add(matK);
    }
  }

  @Override
  public Vec L(Vec at) {
    return VecTools.fill(super.L(at), 1000);
  }

  @Override
  public Mx getContextMat(int idx) {
    final Mx s = contextSymVectors.get(idx);
    final Mx k = contextSkewVectors.get(idx);
    final Mx kkT = VecTools.append(VecTools.outer(k.row(0), k.row(0)), VecTools.outer(k.row(1), k.row(1)));
    final Mx ssT = VecTools.append(VecTools.outer(s.row(0), s.row(0)), VecTools.outer(s.row(1), s.row(1)));
    for (int i = 0; i < kkT.rows(); i++) {
      for (int j = 0; j < i; j++) {
        kkT.set(i, j, kkT.get(i, j) * -1d);
      }
    }
    return VecTools.append(ssT, kkT);
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
    Mx grads = new VecBasedMx(dimX, dimY * 5);

    // Перебираем слова в тексте по очереди, чтобы их обновить вектора
    range(0, text.length()).forEach(pos -> {
      final int word_id = text.at(pos);
      // Для каждого индекса
      range(0, 2).forEach(i ->{
        range(0, dimY).forEach(j -> {
          final Mx dSi = getContextSymMatDerivative(word_id, i, j);
          final Mx dKi = getContextSkewMatDerivative(word_id, i, j);
          double diffS = getContextDerivative(dSi, pos);
          double diffK = getContextDerivative(dKi, pos);
          grads.adjust(word_id, j + i * dimY, diffS);
          grads.adjust(word_id, j + (2 + i) * dimY, diffK);
        });
      });
    });

    Mx C = VecTools.copy(C0);
    for (int pos = 0; pos < text.length(); pos++) {
      final int idx = text.at(pos);
      final Vec im = imageVectors.row(idx);
      final double derivativeTerm = getImageDerivativeTerm(im, C);

      // Для каждого индекса
      for (int j = 0; j < dimY; j++) {
        grads.adjust(idx, j + dimY * 4, getImageDerivative(im, C, derivativeTerm, j));
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
    return grads;
  }

  private Mx getContextSymMatDerivative(int idx, int di, int dj) {
    final Mx s = contextSymVectors.get(idx);
    final Mx result = new VecBasedMx(dimY, dimY);
    VecTools.fill(result, 0d);
    for (int j = 0; j < dimY; j++) {
      result.set(dj, j, s.get(di, j));
      result.set(j, dj, s.get(di, j));
    }
    result.set(dj, dj, 2d * s.get(di, dj));
    return result;
  }

  private Mx getContextSkewMatDerivative(int idx, int di, int dj) {
    final Mx k = contextSkewVectors.get(idx);
    final Mx result = new VecBasedMx(dimY, dimY);
    VecTools.fill(result, 0d);
    for (int j = 0; j < dimY; j++) {
      final int sign = j >= dj ? 1 : -1;
      result.set(dj, j, sign * k.get(di, j));
      result.set(j, dj, -sign * k.get(di, j));
    }
    result.set(dj, dj, 2d * k.get(di, dj));
    return result;
  }

  private void unfold(Vec in) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 5, in);
    for (int i = 0; i < dimX; i++) {
      Mx matS = new VecBasedMx(2, dimY);
      Mx matK = new VecBasedMx(2, dimY);
      for (int j = 0; j < dimY; j++) {
        matS.set(0, j, matrices.get(i, j));
        matS.set(1, j, matrices.get(i, j + dimY));
        matK.set(0, j, matrices.get(i, j + 2 * dimY));
        matK.set(1, j, matrices.get(i, j + 3 * dimY));
        imageVectors.set(i, j, matrices.get(i, j + 4 * dimY));
      }
      contextSymVectors.set(i, matS);
      contextSkewVectors.set(i, matK);
    }
  }

  public static Mx unfoldImages(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 5, in);
    Mx result = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        result.set(i, j, matrices.get(i, j + 4 * dimY));
      }
    }
    return result;
  }

  public static List<Mx> unfoldSymmetricContexts(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 5, in);
    List<Mx> result = new ArrayList<>(dimX);
    for (int i = 0; i < dimX; i++) {
      Mx mat = new VecBasedMx(2, dimY);
      for (int j = 0; j < dimY; j++) {
        mat.set(0, j, matrices.get(i, j ));
        mat.set(1, j, matrices.get(i, j + dimY));
      }
      result.set(i, mat);
    }
    return result;
  }

  public static List<Mx> unfoldSkewsymmetricContexts(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 5, in);
    List<Mx> result = new ArrayList<>(dimX);
    for (int i = 0; i < dimX; i++) {
      Mx mat = new VecBasedMx(2, dimY);
      for (int j = 0; j < dimY; j++) {
        mat.set(0, j, matrices.get(i, j + 2 * dimY));
        mat.set(1, j, matrices.get(i, j + 3 * dimY));
      }
      result.set(i, mat);
    }
    return result;
  }

  public static Mx fold(List<Mx> contextSym, List<Mx> contextSkew, Mx images) {
    int dimX = images.rows();
    int dimY = images.columns();
    Mx res = new VecBasedMx(dimX, dimY * 5);
    for (int i = 0; i < dimX; i++) {
      Mx matS = contextSym.get(i);
      Mx matK = contextSkew.get(i);
      for (int j = 0; j < dimY; j++) {
        res.set(i, j, matS.get(0, j));
        res.set(i, j + dimY, matS.get(1, j));
        res.set(i, j + 2 * dimY, matK.get(0, j));
        res.set(i, j + 3 * dimY, matK.get(1, j));
        res.set(i, j + 4 * dimY, images.get(i, j));
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
      Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * 5, in);
      for (int i = 0; i < dimX; i++) {
        for (int k = 0; k < 2; k++) {
          Vec sym = matrices.row(i).sub(k * dimY, dimY);
          Vec skew = matrices.row(i).sub((2 + k) * dimY, dimY);
          VecTools.normalizeL2(sym);
          VecTools.normalizeL2(skew);
          for (int j = 0; j < dimY; j++) {
            matrices.set(i, j + k * dimY, sym.get(j));
            matrices.set(i, j + (2 + k) * dimY, skew.get(j));
          }
        }
        Vec im;
        im = matrices.row(i).sub(4 * dimY, dimY);
        VecTools.normalizeL2(im);
        for (int j = 0; j < dimY; j++) {
          matrices.set(i, j + dimY * 4, im.get(j));
        }
      }
      return matrices;
      //return in;
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
