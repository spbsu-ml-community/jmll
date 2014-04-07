package com.spbsu.ml.data;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.xml.JDOMUtil;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.data.impl.Hierarchy;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.array.TDoubleArrayList;
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
  public static DataSet loadRegressionAsMC(String file, int classCount, TDoubleArrayList borders)  throws IOException{
    DataSet ds = DataTools.loadFromFeaturesTxt(file);

    double[] target = ds.target().toArray();
    int[] idxs = ArrayTools.sequence(0, target.length);
    ArrayTools.parallelSort(target, idxs);

    if (borders.size() == 0) {
      double min = target[0];
      double max = target[target.length - 1];
      double delta = (max - min) / classCount;
      for (int i = 0; i < classCount; i++) {
        borders.add(delta * (i + 1));
      }
    }

    Vec resultTarget = new ArrayVec(ds.power());
    int targetCursor = 0;
    for (int borderCursor = 0; borderCursor < borders.size(); borderCursor++){
      while (targetCursor < target.length && target[targetCursor] <= borders.get(borderCursor)) {
        resultTarget.set(idxs[targetCursor], borderCursor);
        targetCursor++;
      }
    }
    return new ChangedTarget((DataSetImpl)ds, resultTarget);
  }

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

  public static Hierarchy loadHierarchicalStructure(String hierarchyXml) throws IOException{
    Element catalog = JDOMUtil.loadXML(new File(hierarchyXml));
    Hierarchy.CategoryNode root = traverseLoad(catalog, null);
    return new Hierarchy(root);
  }

  public static void saveHierarchicalStructure(Hierarchy ds, String outFile) throws IOException{
    Hierarchy.CategoryNode root = ds.getRoot();
    Element catalog = traverseSave(root, 0);
    JDOMUtil.flushXML(catalog, new File(outFile));
  }

  private static Element traverseSave(Hierarchy.CategoryNode node, int depth) {
    Element element = new Element("cat" + depth);
    element.setAttribute("id", String.valueOf(node.getCategoryId()));
    element.setAttribute("size", String.valueOf(node.getInnerDS() == null? 0 : node.getInnerDS().power()));
    for (Hierarchy.CategoryNode child : node.getChildren()) {
      element.addContent(traverseSave(child, depth + 1));
    }
    return element;
  }

  private static Hierarchy.CategoryNode traverseLoad(Element element, Hierarchy.CategoryNode parentNode) {
    int id = Integer.parseInt(element.getAttributeValue("id"));
    Hierarchy.CategoryNode node = new Hierarchy.CategoryNode(id, parentNode);
    for (Element child : (List<Element>)element.getChildren())
      node.addChild(traverseLoad(child, node));
    return node;
  }

  public static Hierarchy prepareHierStructForRegressionUniform(int depth)  {
    Hierarchy.CategoryNode root = new Hierarchy.CategoryNode(0, null);

    Hierarchy.CategoryNode[] nodes = new Hierarchy.CategoryNode[(1 << (depth + 1)) - 1];
    nodes[0] = root;
    for (int i = 1; i < nodes.length; i++) {
      Hierarchy.CategoryNode parent = nodes[i % 2 == 0 ? (i - 2) / 2 : (i - 1) / 2];
      nodes[i] = new Hierarchy.CategoryNode(i, parent);
      parent.addChild(nodes[i]);
    }
    return new Hierarchy(root);
  }

  private static class Counter {

    int number = 0;
    public Counter(int init) {this.number = init;}
    public int getNext()     {return number++;}

  }

  public static Hierarchy prepareHierStructForRegressionMedian(Vec targetMC) {
    double[] target = targetMC.toArray();
    int clsCount = DataTools.countClasses(targetMC);
    int[] freq = new int[clsCount];
    for (int i = 0; i < target.length; i++) {
      freq[(int)target[i]]++;
    }
    Hierarchy.CategoryNode root = splitAndAddChildren(freq, 0, freq.length, new Counter(freq.length));
    return new Hierarchy(root);

  }

  private static Hierarchy.CategoryNode splitAndAddChildren(int[] arr, int start, int end, Counter innerNodeIdx) {
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

    Hierarchy.CategoryNode node = new Hierarchy.CategoryNode(innerNodeIdx.getNext(), null);
    if (bestSplit == start) {
      node.addChild(new Hierarchy.CategoryNode(start, node));
    }
    else {
      node.addChild(splitAndAddChildren(arr, start, bestSplit + 1, innerNodeIdx));
    }
    if (bestSplit == end - 2) {
      node.addChild(new Hierarchy.CategoryNode(end - 1, node));
    }
    else {
      node.addChild(splitAndAddChildren(arr, bestSplit + 1, end, innerNodeIdx));
    }
    return node;
  }
}
