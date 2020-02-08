package com.expleague.ml.embedding.LM;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.embedding.decomp.DecompBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class LWMatrixRegression extends FuncC1.Stub {
  protected static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  protected Mx imageVectors;
  protected Mx C0;
  protected int dimX, dimY;
  protected int dimTotal;
  protected IntSeq text;

  public LWMatrixRegression(final int dimx, final int dimy, IntSeq text) {
    this.text = text;
    this.dimX = dimx;
    this.dimY = dimy;
    this.C0 = C0();
    imageVectors = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        imageVectors.set(i, j, initializeValue(dimY));
      }
      //VecTools.normalizeL2(imageVectors.row(i));
    }
  }


  public double value() {
    double probab = 1d;
    Mx C = VecTools.copy(C0);
    for (int t = 0; t < text.length(); t++) {
      final int idx = text.at(t);
      probab *= getProbability(C, idx);
      C = MxTools.multiply(C, getContextMat(idx));
    }
    //System.out.println(imageVectors.toString());
    return probab;
  }


  public double getProbability(Mx C, int image_idx) {
    final Vec im = imageVectors.row(image_idx);
    final double uCu = VecTools.multiply(im, MxTools.multiply(C, im));
    double softSum = 0d;
    for (int i = 0; i < dimX; i++) {
      final Vec img = imageVectors.row(i);
      final double e = Math.exp(-VecTools.multiply(img, MxTools.multiply(C, img)) + uCu);
      /*if (e == POSITIVE_INFINITY) {
        return MIN_PROBAB;
      }*/
      softSum += e;
    }
    return 1d / softSum;
  }


  public abstract Mx getContextMat(int idx);


  protected double getImageDerivativeTerm(final Vec im, final Mx C) {
    double softSum = 0d;
    final double uCu = VecTools.multiply(im, MxTools.multiply(C, im));
    for (int i = 0; i < dimX; i++) {
      final Vec img = imageVectors.row(i);
      final double e = Math.exp(-VecTools.multiply(img, MxTools.multiply(C, img)) + uCu);
      /*if (e == POSITIVE_INFINITY) {
        return 1d - MIN_PROBAB;
      }*/
      softSum += e;
    }
    return 1 - 1d / softSum;
  }

  protected double getImageDerivative(final Vec im, final Mx C, final double derivativeTerm, int i) {
    double diff = 0d;
    for (int j = 0; j < im.dim(); j++) {
      diff += im.get(j) * (C.get(i, j) + C.get(j, i));
    }
    return diff * derivativeTerm;
  }

  protected double getContextDerivative(Mx dContext_pos, int pos) {
    Mx C = VecTools.copy(C0);
    Mx dC = VecTools.copy(C0);
    double diff = 0;

    //for (int t = Math.max(0, pos - window_left); t < Math.min(text.length(), pos + window_right + 1); t++) {
    for (int t = 0; t < text.length(); t++) {
      final int idx = text.at(t);
      final Vec im = imageVectors.row(idx);
      final Mx context = getContextMat(idx);
      if (t < pos) {
        C = MxTools.multiply(C, context);
      } else if (t == pos) {
        dC = VecTools.copy(C);
        C = MxTools.multiply(C, context);
      } else if (t == pos + 1) {
        dC = MxTools.multiply(dC, dContext_pos);
        diff = getContextDerivativeTerm(im, C, dC);
        C = MxTools.multiply(C, context);
      } else {
        dC = MxTools.multiply(dC, context);
        diff += getContextDerivativeTerm(im, C, dC);
        C = MxTools.multiply(C, context);
      }
    }
    return diff;
  }


  private double getContextDerivativeTerm(final Vec im, final Mx C, final Mx dC) {
    double result = VecTools.multiply(im, MxTools.multiply(dC, im));

    for (int i = 0; i < dimX; i++) {
      double softSum = 0d;
      final Vec u = imageVectors.row(i);
      double uCu = VecTools.multiply(u, MxTools.multiply(C, u));
      for (int j = 0; j < dimX; j++) {
        final Vec img = imageVectors.row(j);
        final double e = Math.exp(-VecTools.multiply(img, MxTools.multiply(C, img)) + uCu);
        /*if (e == POSITIVE_INFINITY) {
          uCu = MIN_PROBAB;
          softSum = 1d;
          break;
        }*/
        softSum += e;
      }
      //result += -uCu / softSum;
      result += -VecTools.multiply(u, MxTools.multiply(dC, u)) / softSum;
    }

    return result;
  }


  public Mx C0() {
    final Mx mat = new VecBasedMx(dimY, dimY);
    // Как там единичную матрицу задать функцией?
    VecTools.fill(mat, 0d);
    for (int i = 0; i < dimY; i++) {
      mat.set(i, i, 1d);
    }
    return mat;
  }

  protected double initializeValue(int dim) {
    return (Math.random() - 0.5) / dim / 100.;
  }

}
