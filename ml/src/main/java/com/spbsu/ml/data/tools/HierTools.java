package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.stat.impl.NumericSampleDistribution;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.tree.FastTree;
import com.spbsu.commons.util.tree.Node;
import com.spbsu.commons.util.tree.NodeVisitor;
import com.spbsu.commons.util.tree.Tree;
import com.spbsu.commons.util.tree.impl.node.InternalNode;
import com.spbsu.commons.util.tree.impl.node.LeafNode;
import com.spbsu.commons.xml.JDOMUtil;
import com.spbsu.ml.data.impl.HierarchyTree;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TDoubleList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.procedure.TIntProcedure;
import gnu.trove.stack.array.TIntArrayStack;
import org.jdom.Element;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;

/**
* User: qdeee
* Date: 07.04.14
*/
public class HierTools {
  public static Tree pruneTree(final Tree tree, final IntSeq target, final int minEntries) {
    final TIntIntMap id2count = itemsDeepCounter(tree, target);
    final NodeVisitor cutter = new NodeVisitor() {
      @Override
      public Node visit(final InternalNode node) {
        if (id2count.get(node.id) > minEntries) {
          final List<Node> newChildren = new LinkedList<>();
          for (Node child : node.getChildren()) {
            final Node newChild = (Node) child.accept(this);
            if (newChild != null) {
              newChildren.add(newChild);
            }
          }
          if (newChildren.size() > 1) {
            final InternalNode newNode = new InternalNode(node.id);
            for (Node newChild : newChildren) {
              newNode.addChild(newChild);
            }
            return newNode;
          }
          else {
            return new LeafNode(node.id);
          }
        }
        else {
          return null;
        }
      }

      @Override
      public LeafNode visit(final LeafNode node) {
        if (id2count.get(node.id) > minEntries) {
          return new LeafNode(node.id);
        }
        else {
          return null;
        }
      }
    };

    final InternalNode pruned = (InternalNode) tree.getRoot().accept(cutter);
    return new FastTree(pruned);
  }


  public static void createTreesMapping(final Node from, final Node to, final TIntIntMap map) {
    map.put(from.id, to.id);
    if (from instanceof InternalNode) {
      outer:
      for (Node fromChild : ((InternalNode) from).getChildren()) {
        if (to instanceof InternalNode) {
          for (Node toChild : ((InternalNode) to).getChildren()) {
            if (fromChild.id == toChild.id) {
              createTreesMapping(fromChild, toChild, map);
              continue outer;
            }
          }
        }
        else {
          createTreesMapping(fromChild, to, map);
        }
      }
    }
  }


  public static TIntIntMap itemsDeepCounter(final Tree tree, final IntSeq target) {
    final TIntIntMap id2count = new TIntIntHashMap();
    for (int i = 0; i < target.length(); i++) {
      id2count.adjustOrPutValue(target.at(i), 1, 1);
    }

    final TIntIntMap id2deepCount = new TIntIntHashMap();
    tree.getRoot().accept(new NodeVisitor<Integer>() {
      @Override
      public Integer visit(final InternalNode node) {
        int count = 0;
        for (Node child : node.getChildren()) {
          count += child.accept(this);
        }
        id2deepCount.put(node.id, count);
        return count;
      }

      @Override
      public Integer visit(final LeafNode node) {
        final int count = id2count.get(node.id);
        id2deepCount.put(node.id, count);
        return count;
      }
    });
    return id2deepCount;
  }

  private static class Counter {
    int number = 0;
    public Counter(int init) {this.number = init;}
    public int getNext()     {return number++;}

  }

  public static FastTree loadOrderedMulticlassAsHierarchicalMedian(IntSeq targetMC) {
    final int countClasses = MCTools.countClasses(targetMC);
    final int[] counts = new int[countClasses];
    for (int i = 0; i < targetMC.length(); i++) {
      counts[targetMC.at(i)]++;
    }
    final Deque<Node> nodes = new LinkedList<>();
    final Queue<Pair<Integer, Integer>> borders = new LinkedList<>();

    int newNodesCounter = countClasses;
    borders.add(Pair.create(0, counts.length));
    while (borders.size() > 0) {
      final Pair<Integer, Integer> pop = borders.poll();
      final int from = pop.first;
      final int end = pop.second;

      if (end - from == 1) {
        nodes.add(new LeafNode(from));
      }
      else {
        final int sum = ArrayTools.sum(counts, from, end);

        int bestSplit = -1;
        int minSubtract = Integer.MAX_VALUE;
        int curSum = 0;
        for (int split = from; split < end - 1; split++) {
          curSum += counts[split];
          int subtract = Math.abs((sum - curSum) - curSum);
          if (subtract < minSubtract) {
            minSubtract = subtract;
            bestSplit = split;
          }
        }

        final InternalNode node = new InternalNode(newNodesCounter++);
        nodes.add(node);

        borders.add(Pair.create(from, bestSplit + 1));
        borders.add(Pair.create(bestSplit + 1, end));
      }
    }

    final Stack<Node> tempStack = new Stack<>();
    while (nodes.size() > 1) {
      final Node node2 = nodes.removeLast();
      final Node node1 = nodes.removeLast();
      while (nodes.peekLast() instanceof LeafNode || nodes.peekLast() instanceof InternalNode && ((InternalNode) nodes.peekLast()).getChildren().size() != 0) {
        tempStack.push(nodes.removeLast());
      }
      final InternalNode node = (InternalNode) nodes.peekLast();
      node.addChild(node1);
      node.addChild(node2);
      while (tempStack.size() > 0) {
        nodes.addLast(tempStack.pop());
      }
    }
    return new FastTree(nodes.poll());
  }


  ///////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////

  @Deprecated
  public static FastTree _loadOrderedMulticlassAsHierarchicalMedian(IntSeq targetMC) {
    final int countClasses = MCTools.countClasses(targetMC);
    final int[] counts = new int[countClasses];
    for (int i = 0; i < targetMC.length(); i++) {
      counts[targetMC.at(i)]++;
    }
    final Node root = splitAndCreateChildren(counts, 0, counts.length, new Counter(counts.length));
    return new FastTree(root);
  }

  @Deprecated
  private static Node splitAndCreateChildren(int[] arr, int start, int end, Counter innerNodeIdx) {
    int sum = 0;
    for (int i = start; i < end; i++) {
      sum += arr[i];
    }

    int bestSplit = -1;
    int minSubtract = Integer.MAX_VALUE;
    int curSum = 0;
    for (int split = start; split < end - 1; split++) {
      curSum += arr[split];
      int subtract = Math.abs((sum - curSum) - curSum);
      if (subtract < minSubtract) {
        minSubtract = subtract;
        bestSplit = split;
      }
    }

    final InternalNode node = new InternalNode(innerNodeIdx.getNext());
    if (bestSplit == start) {
      node.addChild(new LeafNode(start));
    }
    else {
      node.addChild(splitAndCreateChildren(arr, start, bestSplit + 1, innerNodeIdx));
    }
    if (bestSplit == end - 2) {
      node.addChild(new LeafNode(end - 1));
    }
    else {
      node.addChild(splitAndCreateChildren(arr, bestSplit + 1, end, innerNodeIdx));
    }
    return node;
  }

  @Deprecated
  public static void convertCatalogXmlToTree(File catalogXml, File out) throws IOException {
    final TIntIntHashMap id2parentId = new TIntIntHashMap();
    final TIntObjectHashMap<Element> id2node = new TIntObjectHashMap<Element>();

    final Element catalog = JDOMUtil.loadXML(catalogXml).getChild("cat");
    final List<Element> items  = catalog.getChildren();
    for (Element item : items) {
      try {
        final int id = Integer.parseInt(item.getAttributeValue("id"));
        if (id != 0) {
          final int parentId = Integer.parseInt(item.getAttributeValue("parent_id"));
          id2parentId.put(id, parentId);
        }
        id2node.put(id, item);
      }
      catch (NumberFormatException e) {
        System.out.println(item.getAttributes().toString());
      }
    }

    for (TIntObjectIterator<Element> iter = id2node.iterator(); iter.hasNext(); ) {
      iter.advance();
      final int id = iter.key();
      if (id != 0) {
        try {
          final int parentId = id2parentId.get(id);//Integer.parseInt(iter.value().getAttributeValue("parent_id"));
          final Element element = iter.value();
          final Element parentElement = id2node.get(parentId);
          if (parentElement != null)
            parentElement.addContent(element.detach());
          else  {
            iter.remove();
            System.out.println("id = " + id + ", unknown parent's id= " + parentId);
          }
        } catch (NumberFormatException e) {
          System.out.println(iter.value().toString());
        }
      }
    }
    JDOMUtil.flushXML(id2node.get(0), out);
  }

  @Deprecated
  public static HierarchyTree loadHierarchicalStructure(String hierarchyXml) throws IOException{
    final Element catalog = JDOMUtil.loadXML(new File(hierarchyXml));
    final HierarchyTree.Node root = traverseLoad(catalog, null);
    return new HierarchyTree(root);
  }

  @Deprecated
  private static HierarchyTree.Node traverseLoad(Element element, HierarchyTree.Node parentNode) {
    final int id = Integer.parseInt(element.getAttributeValue("id"));
    final HierarchyTree.Node node = new HierarchyTree.Node(id, parentNode);
    for (Element child : (List<Element>)element.getChildren())
      node.addChild(traverseLoad(child, node));
    return node;
  }

  @Deprecated
  public static void saveHierarchicalStructure(HierarchyTree ds, String outFile) throws IOException{
    final HierarchyTree.Node root = ds.getRoot();
    final Element catalog = traverseSave(root, 0);
    JDOMUtil.flushXML(catalog, new File(outFile));
  }

  @Deprecated
  private static Element traverseSave(HierarchyTree.Node node, int depth) {
    final Element element = new Element("cat" + depth);
    element.setAttribute("id", String.valueOf(node.getCategoryId()));
    element.setAttribute("size", String.valueOf(node.getEntriesCount()));
    for (HierarchyTree.Node child : node.getChildren()) {
      element.addContent(traverseSave(child, depth + 1));
    }
    return element;
  }

  @Deprecated
  public static HierarchyTree prepareHierStructForRegressionUniform(int depth)  {
    final HierarchyTree.Node root = new HierarchyTree.Node(0, null);

    final HierarchyTree.Node[] nodes = new HierarchyTree.Node[(1 << (depth + 1)) - 1];
    nodes[0] = root;
    for (int i = 1; i < nodes.length; i++) {
      final HierarchyTree.Node parent = nodes[i % 2 == 0 ? (i - 2) / 2 : (i - 1) / 2];
      nodes[i] = new HierarchyTree.Node(i, parent);
      parent.addChild(nodes[i]);
    }
    return new HierarchyTree(root);
  }

  @Deprecated
  public static void printMeanAndVarForClassificationOut(final Vec target, final Vec factor, String comment) {
    final NumericSampleDistribution<Double> distributionPositive = new NumericSampleDistribution<Double>();
    final NumericSampleDistribution<Double> distributionNegative = new NumericSampleDistribution<Double>();
    for (int i = 0; i < factor.dim(); i++) {
      if (target.get(i) > 0)
        distributionPositive.update(factor.get(i));
      else
        distributionNegative.update(factor.get(i));
    }
    System.out.println(comment + " (positive samples), mean = " + distributionPositive.getMean() + ", stddev = " + distributionPositive.getStandardDeviation());
    System.out.println(comment + " (negative samples), mean = " + distributionNegative.getMean() + ", stddev = " + distributionNegative.getStandardDeviation());
  }

  @Deprecated
  public static HierarchyTree prepareHierStructForRegressionMedian(IntSeq targetMC) {
    final int clsCount = MCTools.countClasses(targetMC);
    final int[] freq = new int[clsCount];
    for (int i = 0; i < targetMC.length(); i++) {
      freq[targetMC.at(i)]++;
    }
    final HierarchyTree.Node root = splitAndAddChildren(freq, 0, freq.length, new Counter(freq.length));
    return new HierarchyTree(root);

  }

  @Deprecated
  private static HierarchyTree.Node splitAndAddChildren(int[] arr, int start, int end, Counter innerNodeIdx) {
    int sum = 0;
    for (int i = start; i < end; i++) {
      sum += arr[i];
    }

    int bestSplit = -1;
    int minSubtract = Integer.MAX_VALUE;
    int curSum = 0;
    for (int split = start; split < end - 1; split++) {
      curSum += arr[split];
      int subtract = Math.abs((sum - curSum) - curSum);
      if (subtract < minSubtract) {
        minSubtract = subtract;
        bestSplit = split;
      }
    }

    final HierarchyTree.Node node = new HierarchyTree.Node(innerNodeIdx.getNext(), null);
    if (bestSplit == start) {
      node.addChild(new HierarchyTree.Node(start, node));
    }
    else {
      node.addChild(splitAndAddChildren(arr, start, bestSplit + 1, innerNodeIdx));
    }
    if (bestSplit == end - 2) {
      node.addChild(new HierarchyTree.Node(end - 1, node));
    }
    else {
      node.addChild(splitAndAddChildren(arr, bestSplit + 1, end, innerNodeIdx));
    }
    return node;
  }
}
