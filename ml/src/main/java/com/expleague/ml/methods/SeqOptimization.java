package com.expleague.ml.methods;

import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.TargetFunc;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 11:06
 */
public interface SeqOptimization<T, Loss extends TargetFunc> extends Optimization<Loss, DataSet<Seq<T>>, Seq<T>>{
}
