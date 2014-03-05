package com.spbsu.ml.data.impl;

import com.spbsu.commons.func.CacheHolder;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.ml.data.DSIterator;
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

/**
 * User: qdeee
 * Date: 21.02.14
 */
public class HierDS implements DataSet {
  private DataSet transformedDS;
  private CategoryNode root;

  public HierDS(HierDS.CategoryNode treeRoot) {
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

    transformedDS = new DataSetImpl(
        new VecBasedMx(
            learn.xdim(),
            new IndexTransVec(learn.data(),
                new RowsPermutation(perm, learn.xdim())
            )
        ),
        new IndexTransVec(learn.target(), new ArrayPermutation(perm))
    );
    root.fillDS(transformedDS.data(), id2offset, id2size);
  }

  public HierDS getPrunedCopy(int minEntries) {
    CategoryNode newRoot = root.getPrunedCopy(minEntries);
    return new HierDS(newRoot);
  }

  public static void traversePrint(CategoryNode node) {
    if (node.innerDS == null) {
      System.out.println("Category " + node.categoryId + " is empty.");
    }
    else {
      System.out.println("DataSet of category #" + node.categoryId + " (" + node.innerDS.power() + " entries):");
      for (CategoryNode child : node.children) {
        traversePrint(child);
      }
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

    public TIntList getNonEmptyLabels() {
      TIntList labels = new TIntArrayList(children.size() + 1);
      for (CategoryNode child : children) {
        labels.add(child.categoryId);
      }
      if (DataTools.countClasses(innerDS.target()) == labels.size() + 1)
        labels.add(categoryId);
      return labels;
    }

    private CategoryNode getPrunedCopy(int minEntries) {
      if (isLeaf())
        return new CategoryNode(categoryId, null);

      CategoryNode newNode = new CategoryNode(categoryId, null);
      int bigChilds = 0;
      int lastClassSize = DataTools.classEntriesCount(innerDS.target(), children.size());

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
      int currentClass = 0;
      for (CategoryNode child : children) {
        if (child.innerDS.power() < minEntries) {
          for (int i = 0; i < child.innerDS.power(); i++) {
            lastClassIndexes.add(currentIndex++);
          }
        }
        else {
          for (int i = 0; i < child.innerDS.power(); i++) {
            indexes.add(currentIndex++);
            target.add(currentClass);
          }
          currentClass++;
        }
      }

      while (currentIndex < innerDS.power()) {
        lastClassIndexes.add(currentIndex++);
      }
      if (lastClassIndexes.size() >= minEntries) {
        indexes.addAll(lastClassIndexes);
        target.fill(target.size(), target.size() + lastClassIndexes.size(), currentClass);
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

    private int fillDS(Mx fullData, TIntIntHashMap id2offset, TIntIntHashMap id2size) {
      int nodeSize = id2size.get(categoryId);
      if (isLeaf()) {
        int offset = id2offset.get(categoryId); //0 if not found
        Mx data = fullData.sub(offset, 0, nodeSize, fullData.columns());
        Vec target = VecTools.fill(new ArrayVec(nodeSize), 0.0);
        innerDS = new DataSetImpl(data, target);
        return nodeSize > 0? offset : Integer.MAX_VALUE;
      }
      else {
        int offset = nodeSize > 0? id2offset.get(categoryId) : Integer.MAX_VALUE;
        TDoubleList target = new TDoubleLinkedList();
        int currentClass = 0;
        for (CategoryNode child : children) {
          int childOffset = child.fillDS(fullData, id2offset, id2size);
          int childSize = child.innerDS.power();
          target.fill(target.size(), target.size() + childSize, currentClass++);
          offset = min(offset, childOffset);
        }
        target.fill(target.size(), target.size() + nodeSize, currentClass);

        Mx data = offset < Integer.MAX_VALUE ? fullData.sub(offset, 0, target.size(), fullData.columns())
                                             : new VecBasedMx(0, 0);
        innerDS = new DataSetImpl(data, new ArrayVec(target.toArray()));
        return target.size() > 0? offset : Integer.MAX_VALUE;
      }
    }

    public DataSet getInnerDS() {
      return innerDS;
    }

    public boolean isLeaf() {
      return children.size() == 0;
    }
  }

  public <CH extends CacheHolder, R> R cache(Class<? extends Computable<CH, R>> type) {
    return transformedDS.cache(type);
  }

  public int power() {
    return transformedDS.power();
  }

  public int xdim() {
    return transformedDS.xdim();
  }

  public DSIterator iterator() {
    return transformedDS.iterator();
  }

  public Mx data() {
    return transformedDS.data();
  }

  public Vec target() {
    return transformedDS.target();
  }
}
