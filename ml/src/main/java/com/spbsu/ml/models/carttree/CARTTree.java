package com.spbsu.ml.models.carttree;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Sergey Afonin on 14.03.2017.
 */
public class CARTTree {

  private CARTTreeNode root;
  private int featuresCount;
  private int classesCount;
  private double minEntropy;

  public CARTTree(CARTTreeNode root, int featuresCount, int classesCount) {
    this.root = root;
    this.featuresCount = featuresCount;
    this.classesCount = classesCount;
  }

  public CARTTree(int featuresCount, int classesCount, double minEntropy) {
    this.featuresCount = featuresCount;
    this.classesCount = classesCount;
    this.minEntropy = minEntropy;
  }

  public void fit(List<Vec> data, List<Integer> target) {
    this.root = learn(data, target);
  }

  public int calculate(Vec vec) {
    if (vec.dim() > featuresCount)
      throw new IllegalArgumentException("Dim of vec greater than features count");

    return root.calculate(vec);
  }

  @Override
  public String toString() {
    return root.buildString(new StringBuilder(), "==").toString();
  }

  private CARTTreeNode learn(List<Vec> data, List<Integer> target) {
    Pair<Double, Integer> pair = calcEntropy(target);
    double entropy = pair.getFirst();
    CARTTreeNode bestNode = new CARTTreeNode(pair.getSecond());
    if (entropy <= minEntropy) {
      return bestNode;
    } else {
      for (Vec vec : data) {
        for (int i = 0; i < featuresCount; i++) {
          double border = vec.get(i);
          CARTTreeNode cartTreeNode = new CARTTreeNode(i, border);
          List<Integer> leftTarget = new ArrayList<>();
          List<Integer> rightTarget = new ArrayList<>();
          for (int j = 0; j < data.size(); j++) {
            if (cartTreeNode.predicate(data.get(j))) {
              leftTarget.add(target.get(j));
            } else {
              rightTarget.add(target.get(j));
            }
          }

          if (leftTarget.isEmpty() || rightTarget.isEmpty())
            continue;

          double avgEntropy = (calcEntropy(leftTarget).getFirst() + calcEntropy(rightTarget).getFirst()) / 2;
          if (avgEntropy < entropy) {
            entropy = avgEntropy;
            bestNode = cartTreeNode;
          }
        }
      }
      List<Vec> leftData = new ArrayList<>();
      List<Vec> rightData = new ArrayList<>();
      List<Integer> leftTarget = new ArrayList<>();
      List<Integer> rightTarget = new ArrayList<>();
      for (int j = 0; j < data.size(); j++) {
        if (bestNode.predicate(data.get(j))) {
          leftData.add(data.get(j));
          leftTarget.add(target.get(j));
        } else {
          rightData.add(data.get(j));
          rightTarget.add(target.get(j));
        }
      }
      bestNode.setLeft(learn(leftData, leftTarget));
      bestNode.setRight(learn(rightData, rightTarget));
      return bestNode;
    }
  }


  private Pair<Double, Integer> calcEntropy(List<Integer> target) {
    int totalCount = target.size();
    int[] counts = new int[classesCount];
    for (Integer i : target) {
      counts[i]++;
    }
    double entropy = 0;
    int max = counts[0];
    int listValue = 0;
    for (int i = 0; i < counts.length; i++) {
      if (counts[i] > max) {
        max = counts[i];
        listValue = i;
      }

      double d = counts[i] * 1.0 / totalCount;
      if (d > 0) {
        entropy += -d * Math.log(d);
      }
    }
    return Pair.create(entropy, listValue);
  }

}
