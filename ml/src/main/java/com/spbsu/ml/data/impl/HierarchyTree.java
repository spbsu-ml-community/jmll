package com.spbsu.ml.data.impl;

import java.util.ArrayList;
import java.util.List;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.IndexTransVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

/**
 * User: qdeee
 * Date: 21.02.14
 */
@Deprecated
public class HierarchyTree {
  private final Node root;

  public HierarchyTree(Node treeRoot) {
    this.root = treeRoot;
  }

  public void fill(BlockwiseMLLLogit target) {
    final TIntObjectHashMap<TIntList> label2positions = new TIntObjectHashMap<>();
    final int length = target.labels().length();
    for (int i = 0; i < length; i++) {
      final int label = target.label(i);
      if (label2positions.containsKey(label))
        label2positions.get(label).add(i);
      else {
        final TIntList indexes = new TIntLinkedList();
        indexes.add(i);
        label2positions.put(label, indexes);
      }
    }
    root.fillDS(label2positions);
  }

  public HierarchyTree getPrunedCopy(int minEntries) {
    final Node newRoot = root.getPrunedCopy(minEntries);
    if (newRoot == null || newRoot.classesPositions.size() <= 1) {
      throw new IllegalStateException("Hierarchy is empty after pruning!");
    }
    return new HierarchyTree(newRoot);
  }

  public static TIntIntMap getTargetMapping(Node from, Node to) {
    final TIntIntMap map = new TIntIntHashMap();
    traverseCreateMapping(from, to, map);
    return map;
  }

  private static void traverseCreateMapping(Node from , Node to, TIntIntMap map) {
    map.put(from.categoryId, to.categoryId);
    for (Node fromChild : from.children) {
      boolean hasEqualChild = false;
      for (Node toChild : to.children) {
        if (fromChild.categoryId == toChild.categoryId) {
          traverseCreateMapping(fromChild, toChild, map);
          hasEqualChild = true;
          break;
        }
      }
      if (!hasEqualChild) {
        traverseCreateMapping(fromChild, to, map);
      }
    }
  }

  public static void traversePrint(Node node) {
    if (node.getEntriesCount() == 0) {
      System.out.println("Category " + node.categoryId + " is empty.");
    }
    else {
      System.out.println("DataSet of category #" + node.categoryId + " (" + node.getEntriesCount() + " entries):");
//      DataSet ds = DataTools.getSubset((DataSetImpl) node.sourceDS, node.joinLists().toArray());
//      for (DSIterator iter = ds.iterator(); iter.advance(); ) {
//        System.out.println("y = " + iter.y() + ", x[0] = " + iter.x(0) + ", x[1] = " + iter.x(1));
//      }
    }
    for (Node child : node.children) {
      traversePrint(child);
    }
  }

  public Node getRoot() {
    return root;
  }

  public static class Node {
    int categoryId;
    Node parent;
    List<Node> children;

    TIntObjectMap<TIntList> classesPositions;

    public Node(int categoryId, Node parent) {
      this.categoryId = categoryId;
      this.parent = parent;
      this.children = new ArrayList<>();
      this.classesPositions = new TIntObjectHashMap<>(3);
    }

    public List<Node> getChildren() {
      return children;
    }

    public int getCategoryId() {
      return categoryId;
    }

    public void addChild(Node node) {
      this.children.add(node);
    }

    public void fillDS(final TIntObjectHashMap<TIntList> id2positions) {
      final TIntList positions = id2positions.get(categoryId);
      if (positions != null)
        classesPositions.put(categoryId, new TIntArrayList(positions));
      else
        classesPositions.put(categoryId, new TIntArrayList(0));

      if (!isLeaf()) {
        for (Node child : children) {
          child.fillDS(id2positions);
          classesPositions.put(child.categoryId, child.joinLists());
        }
      }
    }

    private Node getPrunedCopy(int minEntries) {
      if (isLeaf()) {
        final Node node = new Node(categoryId, null);
        node.classesPositions.put(categoryId, new TIntArrayList(0));
        return node;
      }

      final Node node = new Node(categoryId, null);
      final TIntList origLastClass = classesPositions.get(categoryId);
      final TIntList lastClassIdxs = new TIntLinkedList(origLastClass);

      int bigClasses = 0;
      for (Node origChild : children) {
        final TIntList origChildIdxs = classesPositions.get(origChild.categoryId);
        if (origChildIdxs.size() >= minEntries) {
          Node newChild = origChild.getPrunedCopy(minEntries);
          if (newChild != null) {
            newChild.parent = node;
            node.addChild(newChild);
          }
          node.classesPositions.put(origChild.categoryId, new TIntArrayList(origChildIdxs));
          bigClasses++;
        }
        else {
          lastClassIdxs.addAll(origChildIdxs);
        }
      }

      if (lastClassIdxs.size() >= minEntries) {
        node.classesPositions.put(categoryId, lastClassIdxs);
        bigClasses++;
      }
      else
        node.classesPositions.put(categoryId, new TIntArrayList(0));

      if (bigClasses >= 2) {
        return node;
      }
      else {
        if (bigClasses == 1 ) {
          if (node.children.size() > 0 && node.children.get(0).children.size() == 0)
            node.children.clear();
          node.classesPositions.clear();
          node.classesPositions.put(categoryId, new TIntArrayList(0));
          return node;
        }
        else
          return null;
      }
    }

    public boolean isLeaf() {
      return children.size() == 0;
    }

    public boolean isRoot() {
      return parent == null;
    }

    public boolean removeIdx(int index) {
      for (TIntObjectIterator<TIntList> iter = classesPositions.iterator(); iter.hasNext(); ) {
        iter.advance();
        TIntList idxs = iter.value();
        if (idxs.remove(index)) {
          if (iter.key() != categoryId) {
            for (Node child : children) {
              if (child.removeIdx(index))
                break;
            }
          }
          return true;
        }
      }
      return false;
    }

    public BlockwiseMLLLogit createTarget(final TIntArrayList labels, DataSet<?> owner) {
      final TIntList targetList = new TIntLinkedList();
      for (Node child : children) {
        final int categoryId = child.categoryId;
        targetList.fill(targetList.size(), targetList.size() + classesPositions.get(categoryId).size(), categoryId);
      }

      final TIntList selfClass = classesPositions.get(categoryId);
      if (selfClass.size() > 0)
        targetList.fill(targetList.size(), targetList.size() + selfClass.size(), categoryId);
      final IntSeq intTarget = MCTools.normalizeTarget(new IntSeq(targetList.toArray()), labels);
      return new BlockwiseMLLLogit(intTarget, owner);
    }

    private VecDataSet createDS(VecDataSet learn, TIntList removeIdxs) {
      final TIntList join = joinLists();
      join.removeAll(removeIdxs);
      final int[] perm = join.toArray();
      final Mx data = new VecBasedMx(
          learn.xdim(),
          new IndexTransVec(learn.data(),
              new RowsPermutation(
                  perm,
                  learn.xdim()
              )
          )
      );
      return new VecDataSetImpl(data, learn);
    }

    public VecDataSet createDS(VecDataSet source) {
      return createDS(source, new TIntArrayList());
    }

    public VecDataSet createDSForChild(int chosenCatId, VecDataSet ds) {
      if (!isRoot()) {
        final TIntList idxs = new TIntLinkedList();
        idxs.addAll(ArrayTools.sequence(0, ds.length()));
        for (Node child : children) {
          if (child.categoryId != chosenCatId) {
            idxs.removeAll(classesPositions.get(child.categoryId));
          }
        }
        if (categoryId != chosenCatId) {
          idxs.removeAll(classesPositions.get(categoryId));
        }

        final int[] perm = idxs.toArray();
        final Mx data = new VecBasedMx(
            ds.xdim(),
            new IndexTransVec(ds.data(),
                new RowsPermutation(
                    perm,
                    ds.xdim()
                )
            )
        );
        return new VecDataSetImpl(data, ds);
      }
      return ds;
    }

    public LLLogit createTargetForChild(int chosenCatId, BlockwiseMLLLogit target) {
      final double[] targetArr = new double[target.labels().length()];
      ArrayTools.fill(targetArr, 0, targetArr.length, -1.);
      final TIntList chosenClassIdxs = classesPositions.get(chosenCatId);
      for (TIntIterator iter = chosenClassIdxs.iterator(); iter.hasNext(); ) {
        targetArr[iter.next()] = 1.;
      }
      return new LLLogit(new ArrayVec(targetArr), target.owner());
    }

    public boolean hasOwnDS() {
      return classesPositions.get(categoryId).size() > 0;
    }

    public boolean isTrainingNode() {
      return classesPositions.size() > 1;
    }

    public TIntList joinLists() {
      final TIntList result = new TIntLinkedList();
      for (Node child : children) {
        result.addAll(classesPositions.get(child.categoryId));
      }
      result.addAll(classesPositions.get(categoryId));
      return result;
    }

    public int getEntriesCount() {
      int sum = 0;
      for (TIntObjectIterator<TIntList> i = classesPositions.iterator(); i.hasNext();) {
        i.advance();
        sum += i.value().size();
      }
      return sum;
    }

    public void addErrorChild(TIntList idxs) {
      final Node child = new Node(-1, this);
      child.classesPositions.put(-1, new TIntArrayList(0));
      children.add(child);
      classesPositions.put(-1, idxs);
    }

    @Override
    public String toString() {
      return "id=" + categoryId + ", entries count=" + getEntriesCount() + ", children size=" + children.size();
    }
  }
}
