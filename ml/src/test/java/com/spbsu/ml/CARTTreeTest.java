package com.spbsu.ml;

import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.models.carttree.CARTTree;
import com.spbsu.ml.models.carttree.CARTTreeNode;
import org.junit.Test;

/**
 * Created by Sergey Afonin on 14.03.2017.
 */
public class CARTTreeTest {

  @Test
  public void printTest() {
    CARTTreeNode list1 = new CARTTreeNode(1.33,0);
    CARTTreeNode list2 = new CARTTreeNode(1.33,1);
    CARTTreeNode list3 = new CARTTreeNode(1.33,2);
    CARTTreeNode list4 = new CARTTreeNode(1.33,3);
    CARTTreeNode leftNode = new CARTTreeNode(0, 8.);
    leftNode.setLeft(list1);
    leftNode.setRight(list2);
    CARTTreeNode rightNode = new CARTTreeNode(0, 18.);
    rightNode.setLeft(list3);
    rightNode.setRight(list4);
    CARTTreeNode root = new CARTTreeNode(0, 12.);
    root.setLeft(leftNode);
    root.setRight(rightNode);
    CARTTree tree = new CARTTree(root, 2, 4);
    System.out.println(tree);

    System.out.println("Calculate: " + tree.calculate(new ArrayVec(4,5)));
    System.out.println("Calculate: " + tree.calculate(new ArrayVec(9,5)));
    System.out.println("Calculate: " + tree.calculate(new ArrayVec(13,5)));
    System.out.println("Calculate: " + tree.calculate(new ArrayVec(19,5)));
  }
}
