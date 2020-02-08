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

class LWSimpleRegression extends LWMatrixRegression {
  private List<Mx> contextMatrices;

  public LWSimpleRegression(int dimx, int dimy, IntSeq text) {
    super(dimx, dimy, text);
    this.dimTotal = dimX * dimY * dimY + dimX * dimY;

    contextMatrices = new ArrayList<>(dimX);
    for (int i = 0; i < dimX; i++) {
      Mx mat = new VecBasedMx(dimY, dimY);
      for (int j = 0; j < dimY; j++) {
        for (int k = 0; k < dimY; k++) {
          mat.set(j, k, initializeValue(dimY));
        }
      }
      //VecTools.normalizeL2(mat);
      contextMatrices.add(mat);
    }
  }

  @Override
  public Mx getContextMat(int idx) {
    return contextMatrices.get(idx);
  }

  // matconcat: dimX строк, в каждой развертка dimY * dimY контекстной матрицы + dimY образ
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
  public Vec L(Vec at) {
    return VecTools.fill(super.L(at), 10000);
  }

  @Override
  public Vec gradient(Vec in) {
    unfold(in);
    Mx grads = new VecBasedMx(dimX, dimY * (dimY + 1));

    range(0, text.length()).parallel().forEach(pos -> {
      final int word_id = text.at(pos);
      // Для каждого индекса
      range(0, dimY).parallel().forEach(i -> {
        range(0, dimY).parallel().forEach(j -> {
          final Mx dC = getContextMatDerivative(i, j);
          final double diff = getContextDerivative(dC, pos);
          grads.adjust(word_id, i * dimY + j, diff);
        });
      });
    });

    Mx C = VecTools.copy(C0);
    final int shift = dimY * dimY;
    for (int pos = 0; pos < text.length(); pos++) {
      final int idx = text.at(pos);
      final Vec im = imageVectors.row(idx);
      final double derivativeTerm = getImageDerivativeTerm(im, C);

      // Для каждого индекса
      Mx finalC = C;
      range(0, dimY).parallel().forEach(j -> {
        grads.adjust(idx, shift + j, getImageDerivative(im, finalC, derivativeTerm, j));
      });
      C = MxTools.multiply(C, getContextMat(idx));
    }

    /*double sqim1 = VecTools.multiply(imageVectors.row(0), imageVectors.row(0));
    double sqim2 = VecTools.multiply(imageVectors.row(1), imageVectors.row(1));
    double im1C1im1 = VecTools.multiply(imageVectors.row(0), MxTools.multiply(contextMatrices.get(0), imageVectors.row(0)));
    double im2C1im2 = VecTools.multiply(imageVectors.row(1), MxTools.multiply(contextMatrices.get(0), imageVectors.row(1)));
    Mx ct1 = contextMatrices.get(0);
    double dim00 = 2. * imageVectors.get(0, 0)  / (1. + Math.exp(sqim2 - sqim1));
    double dim01 = 2. * imageVectors.get(0, 1)  / (1. + Math.exp(sqim2 - sqim1));
    double dim10 = (2. * imageVectors.get(1, 0) * ct1.get(0, 0) +
        imageVectors.get(1, 1) * (ct1.get(0,1) + ct1.get(1, 0))) / (1. + Math.exp(im1C1im1 - im2C1im2));
    double dim11 = (2. * imageVectors.get(1, 1) * ct1.get(1, 1) +
        imageVectors.get(1, 0) * (ct1.get(0,1) + ct1.get(1, 0))) / (1. + Math.exp(im1C1im1 - im2C1im2));
    double sqim10 = imageVectors.get(0, 0) * imageVectors.get(0, 0);
    double sqim20 = imageVectors.get(1, 0) * imageVectors.get(1, 0);
    double dct100 = sqim20 + (Math.exp(-im1C1im1) * -sqim10 + Math.exp(-im2C1im2) * -sqim20) / (Math.exp(-im1C1im1) + Math.exp(-im2C1im2));*/
    return grads;
  }

  private Mx getContextMatDerivative(int di, int dj) {
    final Mx result = new VecBasedMx(dimY, dimY);
    VecTools.fill(result, 0d);
    result.set(di, dj, 1d);
    return result;
  }

  private void unfold(Vec in) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * (dimY + 1), in);
    for (int i = 0; i < dimX; i++) {
      Mx mat = new VecBasedMx(dimY, dimY);
      for (int j = 0; j < dimY * dimY; j++) {
        mat.set(j / dimY, j % dimY, matrices.get(i, j));
      }
      contextMatrices.set(i, mat);
      for (int j = dimY * dimY; j < dimY * (dimY + 1); j++) {
        imageVectors.set(i, j % dimY, matrices.get(i, j));
      }
    }
    int ui = 0;
  }

  public static Mx unfoldImages(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * (dimY + 1), in);
    Mx result = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = dimY * dimY; j < dimY * (dimY + 1); j++) {
        result.set(i, j % dimY, matrices.get(i, j));
      }
    }
    return result;
  }

  public static List<Mx> unfoldContexts(Vec in, int dimX, int dimY) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * (dimY + 1), in);
    List<Mx> result = new ArrayList<>(dimX);
    for (int i = 0; i < dimX; i++) {
      Mx mat = new VecBasedMx(dimY, dimY);
      for (int j = 0; j < dimY * dimY; j++) {
        mat.set(j / dimY, j % dimY, matrices.get(i, j));
      }
      result.add(mat);
    }
    return result;
  }

  public static Mx fold(List<Mx> contexts, Mx images) {
    int dimX = images.rows();
    int dimY = images.columns();
    Mx res = new VecBasedMx(dimX, dimY * (dimY + 1));
    int shift = dimY * dimY;
    for (int i = 0; i < dimX; i++) {
      Mx context = contexts.get(i);
      for (int m1 = 0; m1 < dimY; m1++) {
        for (int m2 = 0; m2 < dimY; m2++) {
          res.set(i, m1 * dimY + m2, context.get(m1, m2));
        }
      }
      for (int j = 0; j < dimY; j++) {
        res.set(i, shift + j, images.get(i, j));
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
      /*Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * (dimY + 1), in);
      for (int i = 0; i < dimX; i++) {
        Mx mat = new VecBasedMx(dimY, dimY);
        for (int j = 0; j < dimY * dimY; j++) {
          mat.set(j / dimY, j % dimY, matrices.get(i, j));
        }
        VecTools.normalizeL2(mat);
        for (int j = 0; j < dimY * dimY; j++) {
          matrices.set(i, j, mat.get(j / dimY, j % dimY));
        }
        Vec im = new ArrayVec(dimY);
        for (int j = dimY * dimY; j < dimY * (dimY + 1); j++) {
          im.set(j % dimY, matrices.get(i, j));
        }
        VecTools.normalizeL2(im);
        for (int j = dimY * dimY; j < dimY * (dimY + 1); j++) {
          matrices.set(i, j, im.get(j % dimY));
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
      return dimX * dimY * (dimY + 1);
    }
  }
}
