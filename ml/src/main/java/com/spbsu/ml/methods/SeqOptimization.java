package com.spbsu.ml.methods;

import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 11:06
 */
public interface SeqOptimization<Loss extends TargetFunc> extends Optimization<Loss, DataSet<Seq>, Seq>{
}
