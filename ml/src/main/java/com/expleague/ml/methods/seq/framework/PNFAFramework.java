package com.expleague.ml.methods.seq.framework;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;

import java.util.*;
import java.util.function.Function;

/**
 * Created by hrundelb on 02.02.19.
 */
public class PNFAFramework {

  private final DFAModel<Integer> targetModel;
  private final DataSet<Seq<Integer>> dataSet;
  private final Vec target;
  private final Random random;

  public PNFAFramework(int stateCount, int stateDim, Alphabet<Integer> alphabet, int trainSize, int trainDim, Random random) {
    this.random = random;
    this.targetModel = generateTargetModel(stateCount, stateDim, alphabet);
    this.dataSet = generateDataSet(alphabet, trainSize, trainDim);
    this.target = calculateTargetVec(stateDim, trainSize);
    System.out.println("Target:\n" + new VecBasedMx(stateDim, target));
  }


  public Function<Seq<Integer>, Vec> test(SeqOptimization<Integer, WeightedL2> optimization) {
    WeightedL2 loss = new WeightedL2(target, dataSet);
    Function<Seq<Integer>, Vec> model = optimization.fit(dataSet, loss);
    for (int i = 0; i < dataSet.length(); i++) {
      Vec targetVec = targetModel.apply(dataSet.at(i));
      Vec actualVec = model.apply(dataSet.at(i));
      System.out.println("=================" + i + "=================");
      System.out.println("target: " + targetVec + "\n actual: " + actualVec);
    }
    return model;
  }

  public DFAModel<Integer> getTargetModel() {
    return targetModel;
  }

  private Vec calculateTargetVec(int stateDim, int trainSize) {
    Vec target = new ArrayVec(trainSize * stateDim);
    for (int i = 0; i < trainSize; i++) {
      Vec result = targetModel.apply(dataSet.at(i));
      VecTools.append(target.sub(i * stateDim, stateDim), result);
    }
    return target;
  }

  private DataSet<Seq<Integer>> generateDataSet(Alphabet<Integer> alphabet, int trainSize, int trainDim) {
    List<Seq<Integer>> data = new ArrayList<>(trainSize);
    for (int i = 0; i < trainSize; i++) {
      IntSeqBuilder builder = new IntSeqBuilder(trainDim);
      for (int j = 0; j < trainDim; j++) {
        Integer t = alphabet.getT(alphabet.condition(random.nextInt(alphabet.size())));
        builder.add(t);
      }
      IntSeq intSeq = builder.build();
      data.add(intSeq);
    }

    return new ListDataSet<>(data);
  }

  private DFAModel<Integer> generateTargetModel(int stateCount, int stateDim, Alphabet<Integer> alphabet) {
    Map<Integer, Mx> weights = new HashMap<>(alphabet.size());
    for (int i = 0; i < alphabet.size(); i++) {
      Integer t = alphabet.getT(alphabet.condition(i));
      Mx weightMx = generateMx(stateCount);
      weights.put(t, weightMx);
    }
    return new DFAModel<>(stateCount, weights);
  }

  private Mx generateMx(int statesCount) {
    Mx mx = new VecBasedMx(statesCount, new ArrayVec(statesCount * statesCount));
    for (int j = 0; j < statesCount; j++) {
      int index = random.nextInt(statesCount);
      mx.set(index, j, 1);
    }
    return mx;
  }
}
