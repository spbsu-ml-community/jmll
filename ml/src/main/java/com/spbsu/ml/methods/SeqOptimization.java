package com.spbsu.ml.methods;

import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 11:06
 */
public interface SeqOptimization<Loss extends Func> extends Optimization<Loss, DataSet<Seq>, Seq>{
}
