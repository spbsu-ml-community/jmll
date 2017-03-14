package com.spbsu.ml.models.carttree;

import com.spbsu.commons.math.vectors.Vec;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Sergey Afonin on 14.03.2017.
 */
public class CARTTreeNode {

  private int featureNumber;
  private double border;
  private CARTTreeNode left;
  private CARTTreeNode right;
  private boolean isList = false;
  private int listValue;
  private double entropy;

  public CARTTreeNode(int featureNumber, double border) {
    this.featureNumber = featureNumber;
    this.border = border;
  }

  public CARTTreeNode(double entropy, int listValue) {
    this.isList = true;
    this.listValue = listValue;
    this.entropy = entropy;
  }

  public void setLeft(CARTTreeNode left) {
    this.left = left;
  }

  public void setRight(CARTTreeNode right) {
    this.right = right;
  }

  public int calculate(Vec vec) {
    if (isList) {
      return listValue;
    } else {
      if (predicate(vec)) {
        return left.calculate(vec);
      } else {
        return right.calculate(vec);
      }
    }
  }

  public double getEntropy() {
    if (isList)
      return entropy;
    else
      return (left.getEntropy() + right.getEntropy()) / 2;
  }

  private boolean predicate(Vec vec) {
    return vec.get(featureNumber) <= border;
  }

  public static CARTTreeNode makeNode(int featureNumber, double border, List<Vec> data, List<Integer> target, int classesCount) {
    CARTTreeNode cartTreeNode = new CARTTreeNode(featureNumber, border);
    List<Integer> leftTarget = new ArrayList<>();
    List<Integer> rightTarget = new ArrayList<>();
    for (int i = 0; i < data.size(); i++) {
      if (cartTreeNode.predicate(data.get(i))) {
        leftTarget.add(target.get(i));
      } else {
        rightTarget.add(target.get(i));
      }
    }
    CARTTreeNode leftList = makeListNode(leftTarget, classesCount);
    CARTTreeNode rightList = makeListNode(rightTarget, classesCount);
    cartTreeNode.setLeft(leftList);
    cartTreeNode.setRight(rightList);
    return cartTreeNode;
  }

  public static CARTTreeNode makeListNode(List<Integer> target, int classesCount) {
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
      entropy += -d * Math.log(d);
    }
    return new CARTTreeNode(entropy, listValue);
  }

  public StringBuilder buildString(StringBuilder builder, String offset) {
    if (isList) {
      builder = builder.append(offset).append("value = ").append(listValue).append(", entropy = ").append(entropy).append("\n");
      return builder;
    } else {
      builder = builder.append(offset).append("x(").append(featureNumber).append(") <= ").append(border).append("\n");
      builder = left.buildString(builder, offset + "==");
      builder = right.buildString(builder, offset + "==");
      return builder;
    }
  }
}
