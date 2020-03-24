package com.expleague.ml.embedding.lm;

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

import java.util.stream.IntStream;

public abstract class LWMatrixRegression extends FuncC1.Stub {
  protected static final Logger log = LoggerFactory.getLogger(DecompBuilder.class.getName());
  protected Mx imageVectors;
  protected Mx C0;
  protected final int dimX, dimY;
  protected final int windowLeft, windowRight;
  protected final static int NO_WINDOW = -1;

  protected IntSeq text;

  public LWMatrixRegression(IntSeq text, final int dimx, final int dimy) {
    this(text, dimx, dimy, NO_WINDOW, NO_WINDOW);
  }

  public LWMatrixRegression(IntSeq text, final int dimx, final int dimy, final int windowLeft, final int windowRight) {
    this.text = text;
    this.dimX = dimx;
    this.dimY = dimy;
    this.windowLeft = windowLeft;
    this.windowRight = windowRight;
    this.C0 = C0();
    imageVectors = new VecBasedMx(dimX, dimY);
    for (int i = 0; i < dimX; i++) {
      for (int j = 0; j < dimY; j++) {
        imageVectors.set(i, j, initializeValue(dimY));
      }
    }
  }

  public double value() {
    return value(text.toArray());
  }

  public double value(int[] sentenceIndexes) {
    double probab = 1d;
    Mx C = VecTools.copy(C0);
    Mx tmp = new VecBasedMx(dimY, dimY);
    for (int idx : sentenceIndexes) {
      probab *= getProbability(C, idx);
      getContextMat(idx, tmp);
      C = MxTools.multiply(C, tmp);
    }
    //System.out.println(imageVectors.toString());
    return probab;
  }

  @Override
  public Vec gradientTo(Vec in, Vec to) {
    unfold(in);
    Mx grads = to instanceof Mx ? (Mx)to : new VecBasedMx(dim() / dimX, to);
    VecTools.scale(grads, 0);
    fillContextGrad(grads);
    fillImageGrad(grads);

    return grads;
  }

  protected void fillImageGrad(final Mx to) {
    Mx C = VecTools.copy(C0);
    final Mx context = new VecBasedMx(dimY, dimY);
    final int shift = dim() / dimX - dimY;
    for (int pos = 0; pos < text.length(); pos++) {
      final int idx = text.at(pos);
      final Vec im = imageVectors.row(idx);
      final double derivativeTerm = getImageDerivativeTerm(im, C);

      Mx finalC = C;
      IntStream.range(0, dimY).parallel().forEach(j -> {
        to.adjust(idx, shift + j, getImageDerivative(im, finalC, derivativeTerm, j));
      });
      getContextMat(idx, context);
      C = MxTools.multiply(C, context);
    }
  }

  protected abstract void fillContextGrad(final Mx to);

  protected abstract void unfold(Vec in);

  public abstract void getContextMat(int idx, final Mx to);

  public abstract Mx getParameters();

  public double getProbability(Mx C, int image_idx) {
    final Vec im = imageVectors.row(image_idx);
    final double uCu = MxTools.quadraticForm(C, im);
    double softSum = 0d;
    for (int i = 0; i < dimX; i++) {
      final Vec img = imageVectors.row(i);
      final double e = Math.exp(-MxTools.quadraticForm(C, img) + uCu);
      /*if (e == POSITIVE_INFINITY) {
        return MIN_PROBAB;
      }*/
      softSum += e;
    }
    return 1d / softSum;
  }

  public Mx getImageVectors() {
    return imageVectors;
  }


  protected double getImageDerivativeTerm(final Vec im, final Mx C) {
    double softSum = 0d;
    final double uCu = MxTools.quadraticForm(C, im);
    for (int i = 0; i < dimX; i++) {
      final Vec img = imageVectors.row(i);
      final double e = Math.exp(-MxTools.quadraticForm(C, img) + uCu);
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
    final Mx context = new VecBasedMx(dimY, dimY);
    double diff = 0;

    for (int t = Math.max(0, pos - windowLeft); t < Math.min(text.length(), pos + windowRight + 1); t++) {
    //for (int t = 0; t < text.length(); t++) {
      final int idx = text.at(t);
      final Vec im = imageVectors.row(idx);
      getContextMat(idx, context);
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
    double result = MxTools.quadraticForm(dC, im);

    for (int i = 0; i < dimX; i++) {
      double softSum = 0d;
      final Vec u = imageVectors.row(i);
      double uCu = MxTools.quadraticForm(C, u);
      for (int j = 0; j < dimX; j++) {
        final Vec img = imageVectors.row(j);
        final double e = Math.exp(-MxTools.quadraticForm(C, img) + uCu);
        /*if (e == POSITIVE_INFINITY) {
          uCu = MIN_PROBAB;
          softSum = 1d;
          break;
        }*/
        softSum += e;
      }
      //result += -uCu / softSum;
      result += -MxTools.quadraticForm(dC, u) / softSum;
    }

    return result;
  }


  public Mx C0() {
    // Как там единичную матрицу задать функцией?
    return MxTools.E(dimY);
  }

  protected double initializeValue(int dim) {
    return (Math.random() - 0.5) / dim;
  }

}
