package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.stat.impl.NumericSampleDistribution;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.xml.JDOMUtil;
import com.spbsu.ml.data.impl.HierarchyTree;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.jdom.Element;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * User: qdeee
 * Date: 07.04.14
 */
public class HierTools {

  public static void convertCatalogXmlToTree(File catalogXml, File out) throws IOException {
    TIntIntHashMap id2parentId = new TIntIntHashMap();
    TIntObjectHashMap<Element> id2node = new TIntObjectHashMap<Element>();

    Element catalog = JDOMUtil.loadXML(catalogXml).getChild("cat");
    List<Element> items  = catalog.getChildren();
    for (Element item : items) {
      try {
        int id = Integer.parseInt(item.getAttributeValue("id"));
        if (id != 0) {
          int parentId = Integer.parseInt(item.getAttributeValue("parent_id"));
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
      int id = iter.key();
      if (id != 0) {
        try {
          int parentId = id2parentId.get(id);//Integer.parseInt(iter.value().getAttributeValue("parent_id"));
          Element element = iter.value();
          Element parentElement = id2node.get(parentId);
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

  public static HierarchyTree loadHierarchicalStructure(String hierarchyXml) throws IOException{
    Element catalog = JDOMUtil.loadXML(new File(hierarchyXml));
    HierarchyTree.Node root = traverseLoad(catalog, null);
    return new HierarchyTree(root);
  }

  private static HierarchyTree.Node traverseLoad(Element element, HierarchyTree.Node parentNode) {
    int id = Integer.parseInt(element.getAttributeValue("id"));
    HierarchyTree.Node node = new HierarchyTree.Node(id, parentNode);
    for (Element child : (List<Element>)element.getChildren())
      node.addChild(traverseLoad(child, node));
    return node;
  }

  public static void saveHierarchicalStructure(HierarchyTree ds, String outFile) throws IOException{
    HierarchyTree.Node root = ds.getRoot();
    Element catalog = traverseSave(root, 0);
    JDOMUtil.flushXML(catalog, new File(outFile));
  }

  private static Element traverseSave(HierarchyTree.Node node, int depth) {
    Element element = new Element("cat" + depth);
    element.setAttribute("id", String.valueOf(node.getCategoryId()));
    element.setAttribute("size", String.valueOf(node.getEntriesCount()));
    for (HierarchyTree.Node child : node.getChildren()) {
      element.addContent(traverseSave(child, depth + 1));
    }
    return element;
  }

  public static HierarchyTree prepareHierStructForRegressionUniform(int depth)  {
    HierarchyTree.Node root = new HierarchyTree.Node(0, null);

    HierarchyTree.Node[] nodes = new HierarchyTree.Node[(1 << (depth + 1)) - 1];
    nodes[0] = root;
    for (int i = 1; i < nodes.length; i++) {
      HierarchyTree.Node parent = nodes[i % 2 == 0 ? (i - 2) / 2 : (i - 1) / 2];
      nodes[i] = new HierarchyTree.Node(i, parent);
      parent.addChild(nodes[i]);
    }
    return new HierarchyTree(root);
  }

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

  private static class Counter {

    int number = 0;
    public Counter(int init) {this.number = init;}
    public int getNext()     {return number++;}

  }

  public static HierarchyTree prepareHierStructForRegressionMedian(Vec targetMC) {
    double[] target = targetMC.toArray();
    int clsCount = MCTools.countClasses(targetMC);
    int[] freq = new int[clsCount];
    for (int i = 0; i < target.length; i++) {
      freq[(int)target[i]]++;
    }
    HierarchyTree.Node root = splitAndAddChildren(freq, 0, freq.length, new Counter(freq.length));
    return new HierarchyTree(root);

  }

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

    HierarchyTree.Node node = new HierarchyTree.Node(innerNodeIdx.getNext(), null);
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
