package com.spbsu.ml.methods;

import Jama.Matrix;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.models.QuadraticModel;

public class LeastSquaresQuadratic implements MLMethodOrder1 {
    @Override
    public Model fit(DataSet learn, Oracle1 loss) {
        return fit(learn, loss, new ArrayVec(learn.power()));
    }

    @Override
    public Model fit(DataSet learn, Oracle1 loss, Vec start) {
        if (loss.getClass() != L2Loss.class)
            throw new IllegalArgumentException("LSQuadratic can not be applied to loss other than l2");
        final int n = learn.xdim();

        //Create & fill X
        Mx X = new VecBasedMx(n*(n+3)/2 + 1, n*(n+3)/2 + 1);

        //from (0,0) to (n*(n+1)/2 , n*(n+1)/2)
        for (int k = 0; k < n; k++) {
            for (int l = k; l < n; l++) {
                for (int p = k; p < n; p++) {
                    for (int r = (k*n+l <= p*n+p)? p : l; r < n; r++) {
                        double val = 0.0;
                        for (int t = 0; t < learn.power(); t++) {
                            val += learn.data().get(t, k)
                                    * learn.data().get(t, l)
                                    * learn.data().get(t, p)
                                    * learn.data().get(t, r);
                        }

                        int iPos = getN(k, l, n);
                        int jPos = getN(p, r, n);

                        X.set(iPos, jPos, (k == l)? val : 2 * val);
                        X.set(jPos, iPos, (p == r)? val : 2 * val);
                    }
                }
            }
        }

        int offset = n*(n+1)/2;

        //from (n*(n+1)/2 , 0) to (X.rows()-1 , n*(n+1)/2) and corresponding transposed piece
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int l = k; l < n; l++) {
                    double val = 0.0;
                    for (int t = 0; t < learn.power(); t++) {
                        val += learn.data().get(t, i)
                                * learn.data().get(t, k)
                                * learn.data().get(t, l);
                    }

                    int iPos = offset + i;
                    int jPos = getN(k, l, n);

                    X.set(iPos, jPos, (k == l)? val : 2 * val);
                    X.set(jPos, iPos, val);
                }
            }
        }

        //from (n*(n+1)/2 , n*(n+1)/2) to (X.rows()-1, X.cols()-1);
        //from (X.rows()-1 , 0) to (X.rows()-1 , n*(n+1)/2) and corresponding transposed piece
        for (int k = 0; k < n; k++) {
            for (int l = k; l < n; l++) {
                double val = 0.0;
                for (int t = 0; t < learn.power(); t++) {
                    val += learn.data().get(t, k)
                            * learn.data().get(t, l);
                }

                X.set(offset + k, offset + l, val);
                X.set(offset + l, offset + k, val);

                int iPos = X.rows() - 1;
                int jPos = getN(k, l, n);
                X.set(iPos, jPos, (k == l)? val : 2 * val);
                X.set(jPos, iPos, val);
            }
        }

        for (int k = 0; k < n; k++) {
            double val = 0.0;
            for (int t = 0; t < learn.power(); t++) {
                val += learn.data().get(t, k);
            }

            int iPos = X.rows() - 1;
            int jPos = offset + k;

            X.set(iPos, jPos, val);
            X.set(jPos, iPos, val);
        }

        X.set(X.rows() - 1, X.columns() - 1, 1.0);

        //Create & fill Y
        Vec Y = new ArrayVec(X.rows());
        for (int k = 0; k < n; k++) {
            for (int l = k; l < n; l++) {
                double val = 0.0;
                for (int t = 0; t < learn.power(); t++) {
                    val += learn.target().get(t) * learn.data().get(t, k) * learn.data().get(t, l);
                }
                Y.set(getN(k, l, n), val);
            }
        }
        for (int k = 0; k < n; k++) {
            double val = 0.0;
            for (int t = 0; t < learn.power(); t++) {
                val += learn.target().get(t) * learn.data().get(t, k);
            }
            Y.set(k + offset, val);
        }
        Y.set(Y.dim() - 1, VecTools.sum(learn.target()));

        Matrix matrixX = new Matrix(X.toArray(), X.rows());
        Matrix vectorY = new Matrix(Y.toArray(), Y.dim());
        Matrix sol = matrixX.solve(vectorY);
        Vec solution = new ArrayVec(sol.getColumnPackedCopy());
//
//
//        Mx L = new VecBasedMx(X.rows(), X.columns());
//        Mx Q = new VecBasedMx(X.rows(), X.columns());
//        VecTools.householderLQ(X, L, Q);
//
//        Vec solution = VecTools.multiply(VecTools.transpose(Q),VecTools.multiply(VecTools.inverseLTriangle(L), Y));
//        System.out.println("system solution: " + solution.toString());

        Mx M = new VecBasedMx(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int pos = getN(i, j, n);
                M.set(i, j, solution.get(pos));
                M.set(j, i, solution.get(pos));
            }
        }

        Vec b = new ArrayVec(n);
        for (int i = 0; i < n; i++) {
            b.set(i, solution.get(offset + i));
        }
        double c = solution.get(solution.dim() - 1);

        return new QuadraticModel(M, b, c);
    }

    private int getN(int i, int j, int n) {
        return n*(n+1)/2 - (n-i+1)*(n-i)/2 + j-i;
    }
}
