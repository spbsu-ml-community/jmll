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

class BothRegression extends LWMatrixRegression {
  private List<Mx> contextMatrices;
  private Mx contextSymVectors, contextSkewVectors;
  private LWSimpleRegression simpleRegression;
  private LWSksRegresseion sksRegression;

  public BothRegression(int dimx, int dimy, IntSeq text) {
    super(dimx, dimy, text);
    contextMatrices = new ArrayList<>(dimX);
    contextSymVectors = new VecBasedMx(dimX, dimY);
    contextSkewVectors = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      contextMatrices.add(new VecBasedMx(dimY, dimY));
    }

    simpleRegression = new LWSimpleRegression(dimx, dimy, text);
    sksRegression = new LWSksRegresseion(dimx, dimy, text);
  }

  @Override
  public Mx getContextMat(int idx) {
    return null;
  }

  // matconcat: dimX строк, в каждой развертка dimY * dimY контекстной матрицы + dimY образ
  @Override
  public double value(Vec in) {
    unfold(in);
    return 0;
  }

  @Override
  public int dim() {
    return dimX * (dimY * dimY + 2 * dimY + dimY);
  }

  @Override
  public Vec L(Vec at) {
    return VecTools.fill(super.L(at), 10);
  }

  @Override
  public Vec gradient(Vec in) {
    unfold(in);
    Mx gradsSimple = (VecBasedMx) simpleRegression.gradient(LWSimpleRegression.fold(contextMatrices, imageVectors));
    Mx gradsSks = (VecBasedMx) sksRegression.gradient(LWSksRegresseion.fold(contextSymVectors, contextSkewVectors, imageVectors));
    Mx res = new VecBasedMx(dimX, dimY * dimY + 2 * dimY + dimY);
    int shift = dimY * dimY;
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY * dimY; j++) {
        res.set(i, j, gradsSimple.get(i, j));
      }
      for (int j = 0; j < dimY * 3; j++) {
        res.set(i, j + shift, gradsSks.get(i, j));
      }
    }

    int u = 0;

    return res;
  }

  private void unfold(Vec in) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimY * dimY + 2 * dimY + dimY, in);
    int shiftSym = dimY * dimY;
    int shiftSkew = shiftSym + dimY;
    int shiftIm = shiftSkew + dimY;
    for (int i = 0; i < dimX; i++) {
      Mx mat = new VecBasedMx(dimY, dimY);
      for (int j = 0; j < dimY * dimY; j++) {
        mat.set(j / dimY, j % dimY, matrices.get(i, j));
      }
      contextMatrices.set(i, mat);
      for (int j = 0; j < dimY; j++) {
        contextSymVectors.set(i, j, matrices.get(i, j + shiftSym));
        contextSkewVectors.set(i, j, matrices.get(i, j + shiftSkew));
        imageVectors.set(i, j, matrices.get(i, j + shiftIm));
      }
    }
  }

  public static Mx fold(List<Mx> contexts, Mx syms, Mx skews, Mx images) {
    int dimX = images.rows();
    int dimY = images.columns();
    Mx res = new VecBasedMx(dimX, dimY * dimY + 2 * dimY + dimY);
    int shiftSym = dimY * dimY;
    int shiftSkew = shiftSym + dimY;
    int shiftIm = shiftSkew + dimY;
    for (int i = 0; i < dimX; i++) {
      Mx context = contexts.get(i);
      for (int m1 = 0; m1 < dimY; m1++) {
        for (int m2 = 0; m2 < dimY; m2++) {
          res.set(i, m1 * dimY + m2, context.get(m1, m2));
        }
      }
      for (int j = 0; j < dimY; j++) {
        res.set(i, j + shiftSym, syms.get(i, j));
        res.set(i, j + shiftSkew, skews.get(i, j));
        res.set(i, j + shiftIm, images.get(i, j));
      }
    }
    int io = 0;
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
