//package com.expleague.ml;
//
//import com.expleague.commons.func.Converter;
//import com.expleague.commons.math.AnalyticFunc;
//import com.expleague.commons.math.MathTools;
//import com.expleague.commons.math.vectors.Vec;
//import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
//import com.expleague.ml.data.impl.BinarizedDataSet;
//import com.expleague.ml.data.set.DataSet;
//import com.expleague.ml.data.set.VecDataSet;
//import com.expleague.ml.data.stats.OrderByFeature;
//import com.expleague.ml.io.BFGridStringConverter;
//import com.expleague.ml.loss.L2;
//
//import java.util.Arrays;
//
//import static java.lang.Math.exp;
//
///**
// * User: solar
// * Date: 09.11.12
// * Time: 17:56
// */
//public class FuzzyBFGrid extends BFGridImpl {
//  public FuzzyBFGrid(BFGridImpl grid, VecDataSet ds, L2 loss) {
//    super(grid.allRows());
//    final BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarizeTo(grid);
//
//    final OrderByFeature orderByFeature = ds.cache().cache(OrderByFeature.class, VecDataSet.class);
//    for (int f = 0; f < grid.rows(); f++) {
//      final Row row = grid.row(f);
//      final ArrayPermutation order = orderByFeature.orderBy(f);
//      final int[] direct = order.direct();
//      final int[] idx = new int[direct.length];
//      final int[] conditionIndices = new int[row.size()];
//      final byte[] bins = bds.bins(f);
//
//      double prev = Double.MIN_VALUE;
//      int index = 0;
//      for (int i = 0; i < direct.length; i++) {
//        final double v = ds.at(i).get(f);
//        if (v != prev)
//          index++;
//        idx[i] = index;
//        prev = v;
//
//        conditionIndices[bins[direct[i]]] = i;
//      }
//
//
//      for (int i = 0; i < direct.length; i++) {
//      }
//
//      final AnalyticFunc func = new AnalyticFunc.Stub() {
//        @Override
//        public double value(double x) {
//          double result = 0;
//          for (int b = 0; b < row.size(); b++) {
//            double bSum = 0;
//            double bSum2 = 0;
//            double bWeights = 0;
//
//            double nomA = 0;
//            double nobB = 0;
//            double denomA = 0;
//            double demomB = 0;
//
//            for (int i = 0; i < loss.xdim(); i++) {
//
//            }
//
//            final int conditionIndex = conditionIndices[b];
//            for (int i = 0; i < loss.xdim(); i++) {
//              final double p;
//              if (reverse[i] > conditionIndex)
//                p = exp(-x * (reverse[i] - conditionIndex));
//              else
//                p = exp(-x * (conditionIndex - reverse[i]));
//              final double v = loss.get(i);
//              bWeights += p;
//              bSum += p * v;
//              bSum += p * v * v;
//            }
//            result += bSum2 - bSum * bSum / bWeights;
//          }
//          return result;
//        }
//
//        @Override
//        public double gradient(double x) {
//          throw new UnsupportedOperationException();
//        }
//      };
//      MathTools.bisection(func, 0, 1);
//    }
//  }
//}
