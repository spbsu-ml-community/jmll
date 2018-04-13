//package com.expleague.ml.methods.seq;
//
//import com.expleague.commons.math.FuncC1;
//import com.expleague.commons.math.MathTools;
//import com.expleague.commons.math.vectors.Mx;
//import com.expleague.commons.math.vectors.MxTools;
//import com.expleague.commons.math.vectors.Vec;
//import com.expleague.commons.math.vectors.VecTools;
//import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
//import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
//import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
//import com.expleague.commons.seq.IntSeq;
//import com.expleague.commons.seq.Seq;
//import com.expleague.ml.data.set.DataSet;
//import com.expleague.ml.func.FuncEnsemble;
//import com.expleague.ml.loss.LLLogit;
//import com.expleague.ml.loss.WeightedL2;
//import com.expleague.ml.methods.SeqOptimization;
//import com.expleague.ml.optimization.Optimize;
//import gnu.trove.map.TIntIntMap;
//import gnu.trove.map.hash.TIntIntHashMap;
//
//import java.util.*;
//import java.util.function.Function;
//
//public class PNFAClassifier<Type, Loss extends LLLogit>  implements SeqOptimization<Type, Loss>{
//  private final PNFARegressor<Type, WeightedL2> regressor;
//
//  public PNFAClassifier(PNFARegressor<Type, WeightedL2> delegate) {
//    this.regressor = delegate;
//  }
//
//  @Override
//  public Function<Seq<Type>, Vec> fit(final DataSet<Seq<Type>> learn, final Loss loss) {
//
//    return new PNFAModel(params, stateCount, stateDim, addToDiag, lambda);
//  }
//}
