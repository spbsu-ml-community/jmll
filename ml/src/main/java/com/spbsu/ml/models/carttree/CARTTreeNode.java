package com.spbsu.ml.models.carttree;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.Pair;

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

  public CARTTreeNode(int featureNumber, double border) {
    this.featureNumber = featureNumber;
    this.border = border;
  }

  public CARTTreeNode(int listValue) {
    this.isList = true;
    this.listValue = listValue;
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

  public boolean predicate(Vec vec) {
    return vec.get(featureNumber) <= border;
  }

  public StringBuilder buildString(StringBuilder builder, String offset) {
    if (isList) {
      builder = builder.append(offset).append("value = ").append(listValue).append("\n");
      return builder;
    } else {
      builder = builder.append(offset).append("x(").append(featureNumber).append(") <= ").append(border).append("\n");
      builder = left.buildString(builder, offset + "==");
      builder = right.buildString(builder, offset + "==");
      return builder;
    }
  }
}
