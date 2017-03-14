package com.spbsu.ml.models.carttree;

import com.spbsu.commons.math.vectors.Vec;

import java.util.List;
import java.util.Map;

import static com.spbsu.ml.models.carttree.CARTTreeNode.makeListNode;
import static com.spbsu.ml.models.carttree.CARTTreeNode.makeNode;

/**
 * Created by Sergey Afonin on 14.03.2017.
 */
public class CARTTree {

  private CARTTreeNode root;
  private int featuresCount;
  private int classesCount;

  public CARTTree(CARTTreeNode root, int featuresCount, int classesCount) {
    this.root = root;
    this.featuresCount = featuresCount;
    this.classesCount = classesCount;
  }

  public CARTTree(int featuresCount, int classesCount) {
    this.featuresCount = featuresCount;
    this.classesCount = classesCount;
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
    CARTTreeNode listNode = makeListNode(target, classesCount);
    double entropy = listNode.getEntropy();
    CARTTreeNode bestNode = listNode;
    if (entropy < 0.67) {
      return listNode;
    } else {
      for (Vec vec : data) {
        for (int i = 0; i < featuresCount; i++) {
          double border = vec.get(i);
          CARTTreeNode treeNode = makeNode(i, border, data, target, classesCount);
          double avgEntropy = treeNode.getEntropy();
          if (avgEntropy < entropy) {
            entropy = avgEntropy;
            bestNode = treeNode;
          }
        }
      }

    }
    return null;
  }

}
