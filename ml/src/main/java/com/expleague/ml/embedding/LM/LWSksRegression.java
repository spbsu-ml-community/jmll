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

public class LWSksRegression extends LWMatrixRegression {
  private List<Mx> contextSymVectors, contextSkewVectors;
  private final int dimDecomp, dimYTotal, dimTotal;

  public LWSksRegression(IntSeq text, final int dimx, final int dimy, final int dimDecomp) {
    this(text, dimx, dimy, dimDecomp, NO_WINDOW, NO_WINDOW);
  }

  public LWSksRegression(IntSeq text, final int dimx, final int dimy, final int dimDecomp, final int windowLeft, final int windowRight) {
    super(text, dimx, dimy, windowLeft, windowRight);
    this.dimDecomp = dimDecomp;
    this.dimYTotal = dimY + dimY * this.dimDecomp * 2;
    this.dimTotal = dimX * dimYTotal;

    contextSymVectors = new ArrayList<>(dimX);
    contextSkewVectors = new ArrayList<>(dimX);

    for (int k = 0; k < dimX; k++) {
      Mx matS = new VecBasedMx(dimDecomp, dimY);
      Mx matK = new VecBasedMx(dimDecomp, dimY);
      for (int i = 0; i < dimDecomp; i++) {
        for (int j = 0; j < dimY; j++) {
          matS.set(i, j, initializeValue(dimY));
          matK.set(i, j, initializeValue(dimY));
        }
      }
      //VecTools.normalizeL2(contextSymVectors.row(i));
      //VecTools.normalizeL2(contextSkewVectors.row(i));
      contextSymVectors.add(matS);
      contextSkewVectors.add(matK);
    }
  }

  @Override
  public Vec L(Vec at) {
    return VecTools.fill(super.L(at), 10.);
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
  protected void fillContextGrad(final Mx to) {
    final Mx dSi = new VecBasedMx(dimY, dimY);
    final Mx dKi = new VecBasedMx(dimY, dimY);

    range(0, text.length()).forEach(pos -> {
      final int word_id = text.at(pos);
      range(0, dimDecomp).forEach(i ->{
        range(0, dimY).forEach(j -> {
          getContextSymMatDerivative(dSi, word_id, i, j);
          getContextSkewMatDerivative(dKi, word_id, i, j);
          double diffS = getContextDerivative(dSi, pos);
          double diffK = getContextDerivative(dKi, pos);
          to.adjust(word_id, j + i * dimY, diffS);
          to.adjust(word_id, j + (dimDecomp + i) * dimY, diffK);
        });
      });
    });
  }

  @Override
  public void getContextMat(int idx, final Mx to) {
    final Mx s = contextSymVectors.get(idx);
    final Mx k = contextSkewVectors.get(idx);
    final Mx ssT = outerSqMx(s);
    final Mx kkT = outerSqMx(k);
    for (int i = 0; i < to.rows(); i++) {
      for (int j = 0; j < to.columns(); j++) {
        to.set(i, j, j < i ? ssT.get(i, j) - kkT.get(i, j) : ssT.get(i, j) + kkT.get(i, j));
      }
    }
  }

  private Mx outerSqMx(Mx a) {
    Mx to = new VecBasedMx(a.columns(), a.columns());
    final int n = to.rows();
    for (int i = 0; i < n; i++) {
      for (int j = i; j < n; j++) {
        double cell = 0d;
        for (int k = 0; k < a.rows(); k++) {
          cell += a.get(k, i) * a.get(k, j);
        }
        to.set(i, j, cell);
        to.set(j, i, cell);
      }
    }
    return to;
  }

  private void getContextSymMatDerivative(Mx to, int idx, int di, int dj) {
    final Mx m = contextSymVectors.get(idx);
    VecTools.fill(to, 0d);
    for (int k = 0; k < dimY; k++) {
      to.set(dj, k, m.get(di, k));
      to.set(k, dj, m.get(di, k));
    }
    to.set(dj, dj, 2d * m.get(di, dj));
  }

  private void getContextSkewMatDerivative(Mx to, int idx, int di, int dj) {
    final Mx m = contextSkewVectors.get(idx);
    VecTools.fill(to, 0d);
    for (int k = 0; k < dimY; k++) {
      final int sign = k < dj ? -1 : 1;
      to.set(dj, k, sign * m.get(di, k));
      to.set(k, dj, -sign * m.get(di, k));
    }
    to.set(dj, dj, 2d * m.get(di, dj));
  }

  @Override
  public Mx getParameters() {
    Mx res = new VecBasedMx(dimX, dimYTotal);
    for (int k = 0; k < dimX; k++) {
      for (int i = 0; i < dimDecomp; i++) {
        for (int j = 0; j < dimY; j++) {
          res.set(k, j + i * dimY, contextSymVectors.get(k).get(i, j));
          res.set(k, j + (dimDecomp + i) * dimY, contextSkewVectors.get(k).get(i, j));
        }
      }
      final int shift = dimY * dimDecomp * 2;
      for (int j = 0; j < dimY; j++) {
        res.set(k, j + shift, imageVectors.get(k, j));
      }
    }
    return res;
  }

  @Override
  protected void unfold(Vec in) {
    Mx matrices = in instanceof Mx ? (Mx)in : new VecBasedMx(dimYTotal, in);
    final int shift = dimY * dimDecomp * 2;

    for (int k = 0; k < dimX; k++) {
      Mx matS = new VecBasedMx(dimDecomp, dimY);
      Mx matK = new VecBasedMx(dimDecomp, dimY);
      for (int i = 0; i < dimDecomp; i++) {
        for (int j = 0; j < dimY; j++) {
          matS.set(i, j, matrices.get(k, j + i * dimY));
          matK.set(i, j, matrices.get(k, j + (dimDecomp + i) * dimY));
        }
      }
      for (int j = 0; j < dimY; j++) {
        imageVectors.set(k, j, matrices.get(k, j + shift));
      }
      contextSymVectors.set(k, matS);
      contextSkewVectors.set(k, matK);
    }
  }

  public static class LWRegularizer extends RegularizerFunc.Stub {

    private final int dimX, dimY, dimDecomp;

    public LWRegularizer(int dimX, int dimY, int dimDecomp) {
      this.dimX = dimX;
      this.dimY = dimY;
      this.dimDecomp = dimDecomp;
    }

    @Override
    public Vec project(Vec in) {
      return in;
    }

    // SAGADescent вроде его не юзает
    @Override
    public double value(Vec x) {
      return 0d;
    }

    @Override
    public int dim() {
      return dimX * (dimY + dimY * dimDecomp * 2);
    }
  }
}
