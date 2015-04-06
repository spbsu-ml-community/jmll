package com.spbsu.ml.data.cherry;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;

public class CherryPick {
  public CNF.Clause fit(final CherryLoss loss) {
    List<CherryBestHolder> features = new ArrayList<>(100);
    final CherryPointsHolder subset = loss.subset();
    double currentScore = Double.NEGATIVE_INFINITY;
    loss.startClause();
    while (true) {
      final CherryBestHolder bestHolder = new CherryBestHolder();
      subset.visitAll((feature, start, end, added, out) -> {
        if (!feature.empty()) {
          final double score = loss.score(feature, start, end, added, out);
          bestHolder.update(feature, score, start, end);
        }
      });
      if (bestHolder.getScore() <= currentScore + 1e-9)
        break;
      features.add(bestHolder);
      loss.addCondition(bestHolder.getValue(), bestHolder.startBin(), bestHolder.endBin());
      currentScore = bestHolder.getScore();
    }
    loss.endClause();
    return  createClause(features);
  }

  private CNF.Clause createClause(List<CherryBestHolder> features) {
    Collections.sort(features, (first, second) -> {
      int firstIndex = first.getValue().origFIndex;
      int secondIndex = second.getValue().origFIndex;

      if (firstIndex < secondIndex) {
        return -1;
      } else if (firstIndex > secondIndex) {
        return 1;
      } else {
        return Integer.compare(first.startBin(), second.startBin());
      }
    });

    List<CNF.Condition> conditions = new ArrayList<>(features.size());
    for (int i = 0; i < features.size(); ++i) {
      int j = i + 1;
      BFGrid.BFRow row = features.get(i).getValue();
      int findex = row.origFIndex;
      while (j < features.size() && features.get(j).getValue().origFIndex == findex) {
        ++j;
      }
      BitSet used = new BitSet(row.size() + 1);
      for (int k = i; k < j; ++k) {
        final int startBin = features.get(k).startBin();
        final int end = features.get(k).endBin() + 1;
        used.set(startBin, end);
      }
      conditions.add(new CNF.Condition(row, used));
    }
    return new CNF.Clause(features.get(0).getValue().grid(), conditions.toArray(new CNF.Condition[conditions.size()]));
  }
}


