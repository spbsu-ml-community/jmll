package com.spbsu.ml.methods.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxBuilder;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.MxByRowsBuilder;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multilabel.ClassicMultiLabelLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.models.multilabel.MultiLabelSubsetsModel;
import gnu.trove.list.TIntList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.procedure.TObjectIntProcedure;

import java.util.ArrayList;
import java.util.List;

/**
 * User: qdeee
 * Date: 23.03.15
 */
public class MultiLabelSubsetsMulticlass implements VecOptimization<ClassicMultiLabelLoss> {
  private final VecOptimization<BlockwiseMLLLogit> weak;
  private final int minExamplesCount;

  public MultiLabelSubsetsMulticlass(final VecOptimization<BlockwiseMLLLogit> weak, final int minExamplesCount) {
    this.weak = weak;
    this.minExamplesCount = minExamplesCount;
  }

  @Override
  public MultiLabelSubsetsModel fit(final VecDataSet learn, final ClassicMultiLabelLoss multiLabelLoss) {
    final Mx targets = multiLabelLoss.getTargets();

    //build two mappings: uniq_labels(vec) -> classNum(int) and classNum(int) -> uniq_labels(vec)
    final TObjectIntMap<Vec> vec2class = new TObjectIntHashMap<>();
    final Vec newTarget = new ArrayVec(targets.rows());
    for (int i = 0; i < targets.rows(); i++) {
      final Vec row = targets.row(i);
      final int classNumber = vec2class.adjustOrPutValue(row, 0, vec2class.size());
      newTarget.set(i, classNumber);
    }
    final Vec[] class2vec = new Vec[vec2class.size()];
    vec2class.forEachEntry(new TObjectIntProcedure<Vec>() {
      @Override
      public boolean execute(final Vec labels, final int classNumber) {
        class2vec[classNumber] = labels;
        return true;
      }
    });


    //filter rare labels combinations
    final VecBuilder targetBuilder = new VecBuilder();
    final MxBuilder mxBuilder = new MxByRowsBuilder();
    final List<Vec> filteredClass2Vec = new ArrayList<>();
    final TIntObjectMap<TIntList> classesIdxs = MCTools.splitClassesIdxs(VecTools.toIntSeq(newTarget));
    for (int clazz = 0, normalizedClass = 0; clazz < classesIdxs.size(); clazz++) {
      final TIntList indexes = classesIdxs.get(clazz);
      if (indexes.size() > minExamplesCount) {
        for (int i = 0; i < indexes.size(); i++) {
          targetBuilder.append(normalizedClass);
          mxBuilder.add(learn.at(i));
        }
        filteredClass2Vec.add(class2vec[clazz]);
        normalizedClass++;
      }
    }


    //fit model
    final VecDataSet filteredDs = new VecDataSetImpl(mxBuilder.build(), learn);
    final BlockwiseMLLLogit mllLogit = new BlockwiseMLLLogit(targetBuilder.build(), learn);
    final MCModel model = (MCModel) weak.fit(filteredDs, mllLogit);
    return new MultiLabelSubsetsModel(model, filteredClass2Vec.toArray(new Vec[filteredClass2Vec.size()]));
  }
}
