package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.models.carttree.CARTTree;
import com.spbsu.ml.models.carttree.CARTTreeNode;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * Created by Sergey Afonin on 14.03.2017.
 */
public class CARTTreeTest {

  @Test
  public void printTest() {
    CARTTreeNode list1 = new CARTTreeNode(0);
    CARTTreeNode list2 = new CARTTreeNode(1);
    CARTTreeNode list3 = new CARTTreeNode(2);
    CARTTreeNode list4 = new CARTTreeNode(3);
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

  @Test
  public void simpleTest() {
    List<Vec> data = new ArrayList<>();
    for (double i = 0; i < 20; i++) {
      data.add(new ArrayVec(i));
    }
    List<Integer> target = new ArrayList<>();
    target.add(0);
    target.add(1);
    target.add(1);
    target.add(1);
    target.add(1);
    target.add(0);
    target.add(0);
    target.add(0);
    target.add(0);
    target.add(1);
    target.add(1);
    target.add(1);
    target.add(1);
    target.add(0);
    target.add(0);
    target.add(0);
    target.add(0);
    target.add(0);
    target.add(0);
    target.add(1);

    CARTTree cartTree = new CARTTree(1, 2, 0);
    cartTree.fit(data, target);
    System.out.println(cartTree);

    for (int i = 0; i < data.size(); i++) {
      assertTrue(cartTree.calculate(data.get(i)) == target.get(i));
    }
  }
}
