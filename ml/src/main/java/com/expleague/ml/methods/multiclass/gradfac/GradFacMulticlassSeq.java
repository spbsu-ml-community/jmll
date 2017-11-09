package com.expleague.ml.methods.multiclass.gradfac;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.Pair;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.MultiClassOneVsRest;

public class GradFacMulticlassSeq<T> implements SeqOptimization<T, L2> {
  private final SeqOptimization<T, L2> inner;
  private final Factorization matrixDecomposition;
  private final Class<? extends L2> local;
  private final boolean printErrors;

  public GradFacMulticlassSeq(final SeqOptimization<T, L2> inner, final Factorization matrixDecomposition, final Class<? extends L2> local) {
    this(inner, matrixDecomposition, local, false);
  }

  public GradFacMulticlassSeq(final SeqOptimization<T, L2> inner, final Factorization matrixDecomposition, final Class<? extends L2> local, final boolean printErrors) {
    this.inner = inner;
    this.matrixDecomposition = matrixDecomposition;
    this.local = local;
    this.printErrors = printErrors;
  }

  @Override
  public Computable<Seq<T>, Vec> fit(DataSet<Seq<T>> learn, L2 mllLogitGradient) {
    final Mx gradient = mllLogitGradient.target instanceof Mx
        ? (Mx)mllLogitGradient.target
        : new VecBasedMx(mllLogitGradient.target.dim() / learn.length(), mllLogitGradient.target);
    final Pair<Vec, Vec> pair = matrixDecomposition.factorize(gradient);

    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();

    final double normB = VecTools.norm(b);
    VecTools.scale(b, 1 / normB);
    VecTools.scale(h, normB);

    final L2 loss = DataTools.newTarget(local, h, learn);
    final Computable<Seq<T>, Vec> model = inner.fit(learn, loss);
    return seq -> {
      final Vec b1 = VecTools.copy(b);
      final double res = model.compute(seq).get(0);
      VecTools.scale(b1, res);
      return b1;
    };

//    if (printErrors) {
//      final Mx mxAfterFactor = VecTools.outer(h, b);
//      final Mx mxAfterFit = resultModel.transAll(learn.data());
//      final double gradNorm = VecTools.norm(gradient);
//      final double error1 = VecTools.distance(gradient, mxAfterFactor);
//      final double error2 = VecTools.distance(mxAfterFactor, mxAfterFit);
//      final double totalError = VecTools.distance(gradient, mxAfterFit);
//
//      System.out.println(String.format("grad_norm = %f, err1 = %f, err2 = %f, absErr = %f", gradNorm, error1, error2, totalError));
//    }
//
//    return resultModel; //not MultiClassModel, for boosting compatibility
  }
}


// cn \t gradnorm \t rel_fact_err
/*
if (printErrors) {
      final RealMatrix realMatrix = new Array2DRowRealMatrix(gradient.rows(), gradient.columns());
      final int rows = gradient.rows();
      final int columns = gradient.columns();
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
          realMatrix.setEntry(i, j, gradient.get(i, j));
        }
      }
      final SingularValueDecomposition singularValueDecomposition = new SingularValueDecomposition(realMatrix);
      System.out.print(singularValueDecomposition.getConditionNumber() + "\t");


      final Mx mxAfterFactor = VecTools.outer(h, b);
//      final Mx mxAfterFit = resultModel.transAll(learn.data());
      final double gradNorm = VecTools.norm(gradient);
      final double error1 = VecTools.distance(gradient, mxAfterFactor);
//      final double error2 = VecTools.distance(mxAfterFactor, mxAfterFit);
//      final double totalError = VecTools.distance(gradient, mxAfterFit);

//      System.out.println(String.format("%f\t%f\t%f\t%f", gradNorm, error1, error2, totalError));
      System.out.print(gradNorm + "\t");
      System.out.print(error1 / gradNorm + "\n");
    }

*/