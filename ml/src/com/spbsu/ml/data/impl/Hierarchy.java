package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.min;
import static java.lang.Math.signum;

/**
 * User: qdeee
 * Date: 21.02.14
 */
public class Hierarchy {
  private CategoryNode root;

  public Hierarchy(Hierarchy.CategoryNode treeRoot) {
    this.root = treeRoot;
  }

  public void fill(DataSet learn) {
    TIntObjectHashMap<TIntList> id2positions = new TIntObjectHashMap<TIntList>();
    Vec target = learn.target();
    for (int i = 0; i < target.dim(); i++) {
      int catId = (int) target.get(i);
      if (id2positions.containsKey(catId))
        id2positions.get(catId).add(i);
      else {
        TIntList indexes = new TIntLinkedList();
        indexes.add(i);
        id2positions.put(catId, indexes);
      }
    }

    TIntList nodesOrder = getPostorderedVisitOrder(getRoot(), false);
    TIntIntHashMap id2offset = new TIntIntHashMap(nodesOrder.size());
    TIntIntHashMap id2size = new TIntIntHashMap(nodesOrder.size());

    TIntList idxs = new TIntArrayList(learn.power());

    for (int i = 0; i < nodesOrder.size(); i++) {
      int id = nodesOrder.get(i);
      TIntList catPositions = id2positions.get(id);
      if (catPositions == null) {
        System.out.println("Category " + id + " is skipped: there are no entries in your dataset");
        continue;
      }
      id2offset.put(id, idxs.size());
      id2size.put(id, catPositions.size());
      idxs.addAll(catPositions);
    }

    int[] perm = idxs.toArray();

    DataSet transformedDS = new DataSetImpl(
        new VecBasedMx(
            learn.xdim(),
            new IndexTransVec(learn.data(),
                new RowsPermutation(perm, learn.xdim())
            )
        ),
        new IndexTransVec(learn.target(), new ArrayPermutation(perm))
    );
    root.fillDS(transformedDS, id2offset, id2size);
  }

  public Hierarchy getPrunedCopy(int minEntries) {
    CategoryNode newRoot = root.getPrunedCopy(minEntries);
    return new Hierarchy(newRoot);
  }

  public static TIntIntHashMap getTargetMapping(CategoryNode from, CategoryNode to) {
    TIntIntHashMap map = new TIntIntHashMap();
    traverseCreateMapping(from, to, map);
    return map;
  }

  private static void traverseCreateMapping(CategoryNode from , CategoryNode to, TIntIntHashMap map) {
    map.put(from.categoryId, to.categoryId);
    for (CategoryNode fromChild : from.children) {
      boolean hasEqualChild = false;
      for (CategoryNode toChild : to.children) {
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

  public static void traversePrint(CategoryNode node) {
    if (node.innerDS == null) {
      System.out.println("Category " + node.categoryId + " is empty.");
    }
    else {
      System.out.println("DataSet of category #" + node.categoryId + " (" + node.innerDS.power() + " entries):");
//      for (DSIterator iter = node.innerDS.iterator(); iter.advance(); ) {
//        System.out.println("y = " + iter.y() + ", x[0] = " + iter.x(0) + ", x[1] = " + iter.x(1));
//      }
    }
    for (CategoryNode child : node.children) {
      traversePrint(child);
    }
  }

  public static TIntList getPostorderedVisitOrder(CategoryNode node, boolean noLeafs) {
    TIntList order = new TIntLinkedList();
    for (CategoryNode child : node.children) {
      if (child.isLeaf()) {
        if (!noLeafs)
          order.add(child.categoryId);
      }
      else
        order.addAll(getPostorderedVisitOrder(child, noLeafs));
    }
    order.add(node.categoryId);
    return order;
  }

  public int nodesCount(CategoryNode node) {
    int result = 0;
    for (CategoryNode child : node.children) {
      result += nodesCount(child);
    }
    return result + 1;
  }

  public CategoryNode getRoot() {
    return root;
  }

  public static class CategoryNode {
    CategoryNode parent;
    List<CategoryNode> children;

    int categoryId;
    DataSet innerDS = null;

    public CategoryNode(int categoryId, CategoryNode parent) {
      this.categoryId = categoryId;
      this.parent = parent;
      this.children = new ArrayList<CategoryNode>();
    }

    public List<CategoryNode> getChildren() {
      return children;
    }

    public int getCategoryId() {
      return categoryId;
    }

    public void addChild(CategoryNode node) {
      this.children.add(node);
    }

    private CategoryNode getPrunedCopy(int minEntries) {
      if (isLeaf())
        return new CategoryNode(categoryId, null);

      CategoryNode newNode = new CategoryNode(categoryId, null);
      int bigChilds = 0;
      int lastClassSize = DataTools.classEntriesCount(innerDS.target(), categoryId);
      for (CategoryNode child : children) {
        if (child.innerDS.power() >= minEntries) {
          CategoryNode newChild = child.getPrunedCopy(minEntries);
          if (newChild != null) {
            newChild.parent = newNode;
            newNode.addChild(newChild);
          }
          bigChilds++;
        }
        else {
          lastClassSize += child.innerDS.power();
        }
      }

      if (lastClassSize >= minEntries)
        bigChilds++;

      if (bigChilds >= 2) {
        newNode.innerDS = getPrunedDataSet(minEntries);
        return newNode;
      }
      else {
        if (bigChilds == 1 ) {
          newNode.innerDS = null;
          if (newNode.children.size() == 1 && newNode.children.get(0).children.size() == 0)
            newNode.children.clear();
          return newNode;
        }
        else
          return null;
      }
    }

    private DataSet getPrunedDataSet(int minEntries) {
      TIntList indexes = new TIntLinkedList();
      TIntList lastClassIndexes = new TIntLinkedList();
      TDoubleList target = new TDoubleLinkedList();

      int currentIndex = 0;
      for (CategoryNode child : children) {
        if (child.innerDS.power() < minEntries) {
          for (int i = 0; i < child.innerDS.power(); i++) {
            lastClassIndexes.add(currentIndex++);
          }
        }
        else {
          for (int i = 0; i < child.innerDS.power(); i++) {
            indexes.add(currentIndex++);
            target.add(child.categoryId);
          }
        }
      }

      while (currentIndex < innerDS.power()) {
        lastClassIndexes.add(currentIndex++);
      }
      if (lastClassIndexes.size() >= minEntries) {
        indexes.addAll(lastClassIndexes);
        target.fill(target.size(), target.size() + lastClassIndexes.size(), categoryId);
      }

      Mx data = new VecBasedMx(
          innerDS.data().columns(),
          new IndexTransVec(
              innerDS.data(),
              new RowsPermutation(indexes.toArray(), innerDS.data().columns())
          )
      );
      return new DataSetImpl(data, new ArrayVec(target.toArray()));
    }

    //return offset
    private int fillDS(DataSet ds, TIntIntHashMap id2offset, TIntIntHashMap id2size) {
      int nodeSize = id2size.get(categoryId);
      if (isLeaf()) {
        int offset = id2offset.get(categoryId); //0 if not found
        Mx data = ds.data().sub(offset, 0, nodeSize, ds.data().columns());
        Vec target = ds.target().sub(offset, nodeSize);
        innerDS = new DataSetImpl(data, target);
        return nodeSize > 0? offset : Integer.MAX_VALUE;
      }
      else {
        int offset = nodeSize > 0? id2offset.get(categoryId) : Integer.MAX_VALUE;
        TDoubleList target = new TDoubleLinkedList();
        for (CategoryNode child : children) {
          int childOffset = child.fillDS(ds, id2offset, id2size);
          offset = min(offset, childOffset);
          target.fill(target.size(), target.size() + child.innerDS.power(), child.categoryId);
        }
        target.fill(target.size(), target.size() + nodeSize, categoryId);
        Mx data = ds.data().sub(offset, 0, target.size(), ds.data().columns());
        innerDS = new DataSetImpl(data, new ArrayVec(target.toArray()));
        return offset; //if innerDS.power() == 0 then Integer.MAX_VALUE will be returned
      }
    }

    public Vec normalizeTarget(TIntList labels) {
      if (innerDS == null)
        return null;

      Vec oldTarget = innerDS.target();
      int currentClass = 0;
      Vec result = new ArrayVec(oldTarget.dim());
      TIntIntHashMap map = new TIntIntHashMap();
      for (int i = 0; i < oldTarget.dim(); i++) {
        int oldTargetVal = (int) oldTarget.get(i);
        if (!map.containsKey(oldTargetVal)) {
          map.put(oldTargetVal, currentClass++);
          labels.add(oldTargetVal);
        }
        result.set(i, map.get(oldTargetVal));
      }
      return result;
    }

    public DataSet getInnerDS() {
      return innerDS;
    }

    public boolean isLeaf() {
      return children.size() == 0;
    }
  }
}
