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
  protected final int dimTotal;

  public LWSimpleRegression(IntSeq text, final int dimx, final int dimy) {
    this(text, dimx, dimy, NO_WINDOW, NO_WINDOW);
  }

  public LWSimpleRegression(IntSeq text, final int dimx, final int dimy, final int windowLeft, final int windowRight) {
    super(text, dimx, dimy, windowLeft, windowRight);
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
    return VecTools.fill(super.L(at), 10.);
  }

  @Override
  protected void fillContextGrad(Mx to) {
    range(0, text.length()).parallel().forEach(pos -> {
      final int word_id = text.at(pos);
      range(0, dimY).parallel().forEach(i -> {
        range(0, dimY).parallel().forEach(j -> {
          final Mx dC = getContextMatDerivative(i, j);
          final double diff = getContextDerivative(dC, pos);
          to.adjust(word_id, i * dimY + j, diff);
        });
      });
    });
  }

  @Override
  public void getContextMat(int idx, final Mx to) {
    VecTools.copyTo(contextMatrices.get(idx), to, 0);
  }

  private Mx getContextMatDerivative(int di, int dj) {
    final Mx result = new VecBasedMx(dimY, dimY);
    VecTools.fill(result, 0d);
    result.set(di, dj, 1d);
    return result;
  }

  @Override
  public Mx getParameters() {
    Mx res = new VecBasedMx(dimX, dimY * (dimY + 1));
    int shift = dimY * dimY;
    for (int i = 0; i < dimX; i++) {
      Mx context = contextMatrices.get(i);
      for (int m1 = 0; m1 < dimY; m1++) {
        for (int m2 = 0; m2 < dimY; m2++) {
          res.set(i, m1 * dimY + m2, context.get(m1, m2));
        }
      }
      for (int j = 0; j < dimY; j++) {
        res.set(i, shift + j, imageVectors.get(i, j));
      }
    }
    return res;
  }

  @Override
  protected void unfold(Vec in) {
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
