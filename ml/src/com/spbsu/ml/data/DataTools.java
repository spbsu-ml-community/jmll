package com.spbsu.ml.data;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.RandomExt;
import com.spbsu.commons.text.CharSequenceTools;
import com.spbsu.commons.xml.JDOMUtil;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.CompositeTrans;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.func.TransJoin;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.ObliviousMultiClassTree;
import com.spbsu.ml.models.ObliviousTree;

import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.jdom.Element;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
import java.lang.reflect.Constructor;
import java.util.*;
import java.util.zip.GZIPInputStream;

import static java.lang.Math.max;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:05
 */
public class DataTools {
  public static DataSet loadFromFeaturesTxt(String file) throws IOException {
    return loadFromFeaturesTxt(file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file), null);
  }

  public static DataSet loadFromFeaturesTxt(String file, TIntObjectHashMap<CharSequence> meta) throws IOException {
    return loadFromFeaturesTxt(file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file), meta);
  }

  public static DataSet loadFromFeaturesTxt(Reader reader) throws IOException {
    return loadFromFeaturesTxt(reader, null);
  }

  public static DataSet loadFromFeaturesTxt(Reader in, TIntObjectHashMap<CharSequence> meta) throws IOException {
    final LineNumberReader reader = new LineNumberReader(in);
    List<double[]> set = new LinkedList<double[]>();
    List<Double> targets = new LinkedList<Double>();
    int maxFeatures = 0;
    String line;
    final List<Double> featuresA = new ArrayList<Double>();
    int lindex = 0;
    while ((line = reader.readLine()) != null) {
      try {
        final StringBuffer metaline = new StringBuffer();
        featuresA.clear();
        StringTokenizer tok = new StringTokenizer(line, "\t");
        metaline.append(tok.nextToken()); // group
        targets.add(Double.parseDouble(tok.nextToken()));
        metaline.append("\t").append(tok.nextToken()); // item name
        metaline.append("\t").append(tok.nextToken()); // equality class inside group
        while (tok.hasMoreTokens()) {
          featuresA.add(Double.parseDouble(tok.nextToken()));
        }
        maxFeatures = Math.max(maxFeatures, featuresA.size());
        double[] features = new double[maxFeatures];
        for (int i = 0; i < featuresA.size(); i++) {
          features[i] = featuresA.get(i);
        }
        if (meta != null)
          meta.put(set.size(), metaline);
        set.add(features);
      }
      catch (Throwable th) {
        System.out.println("Failed to parse line " + lindex + ":");
        System.out.println(line);
      }
      lindex++;
    }
    double[] data = new double[maxFeatures * set.size()];
    double[] target = new double[set.size()];
    Iterator<double[]> iterF = set.iterator();
    Iterator<Double> iterT = targets.iterator();
    int featuresCount = maxFeatures;
    int index = 0;
    while (iterF.hasNext()) {
      final double[] features = iterF.next();
      System.arraycopy(features, 0, data, index * featuresCount, features.length);
      target[index] = iterT.next();
      index++;
    }
    return new DataSetImpl(data, target);
  }

  public static void writeModel(Trans result, File to, ModelsSerializationRepository serializationRepository) throws IOException {
    BFGrid grid = grid(result);
    StreamTools.writeChars(CharSequenceTools.concat(result.getClass().getCanonicalName(), "\t", Boolean.toString(grid != null), "\n",
                           serializationRepository.write(result)), to);
  }

  public static Trans readModel(String fileName, ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    final LineNumberReader modelReader = new LineNumberReader(new InputStreamReader(new FileInputStream(fileName)));
    String line = modelReader.readLine();
    CharSequence[] parts = CharSequenceTools.split(line, '\t');
    Class<? extends Trans> modelClazz = (Class<? extends Trans>)Class.forName(parts[0].toString());
    return serializationRepository.read(StreamTools.readReader(modelReader), modelClazz);
  }

  public static BFGrid grid(Trans result) {
    if (result instanceof CompositeTrans) {
      final CompositeTrans composite = (CompositeTrans) result;
      BFGrid grid = grid(composite.f);
      grid = grid == null ? grid(composite.g) : grid;
      return grid;
    }
    else if (result instanceof FuncJoin) {
      final FuncJoin join = (FuncJoin) result;
      for (Func dir : join.dirs()) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    }
    else if (result instanceof TransJoin) {
      final TransJoin join = (TransJoin) result;
      for (Trans dir : join.dirs) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    }
    else if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      for (Trans dir : ensemble.models) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    }
    else if (result instanceof ObliviousTree)
      return ((ObliviousTree)result).grid();
    if (result instanceof ObliviousMultiClassTree)
      return ((ObliviousMultiClassTree)result).binaryClassifier().grid();
    return null;
  }

  public static int countClasses(Vec target) {
    int classesCount = 0;
    for (int i = 0; i < target.dim(); i++) {
      classesCount = max((int)target.get(i) + 1, classesCount);
    }
    return classesCount;
  }

  public static int classEntriesCount(Vec target, int classNo) {
    int result = 0;
    for (int i = 0; i < target.dim(); i++) {
      if ((int)target.get(i) == classNo)
        result++;
    }
    return result;
  }

  public static int[] extractClass(Vec target, int classNo) {
    final TIntArrayList points = new TIntArrayList(target.dim());
    for (int i = 0; i < target.dim(); i++) {
      if (target.get(i) == classNo)
        points.add(i);
    }
    return points.toArray();
  }

  public static DataSet normalizeClasses(DataSet learn) {
    final DSIterator it = learn.iterator();
    final Vec normalized = new ArrayVec(learn.power());
    while (it.advance()) {
      normalized.set(it.index(), normalizeRelevance(it.y()));
    }
    return new ChangedTarget((DataSetImpl)learn, normalized);
  }

  private static <LocalLoss extends L2> double normalizeRelevance(double y) {
    if (y < 0.07)
      return 0.;
    else if (y < 0.14)
      return 1.;
    else if (y < 0.41)
      return 2.;
    else if (y < 0.61)
      return 3.;
    else
      return 4.;
//    return 1.;
  }


  public static Vec value(Mx ds, Func f) {
    Vec result = new ArrayVec(ds.rows());
    for (int i = 0; i < ds.rows(); i++) {
      result.set(i, f.value(ds.row(i)));
    }
    return result;
  }

  public static <LocalLoss extends StatBasedLoss> WeightedLoss<LocalLoss> bootstrap(LocalLoss loss, RandomExt rnd) {
    int[] poissonWeights = new int[loss.xdim()];
    for (int i = 0; i < loss.xdim(); i++) {
      poissonWeights[i] = rnd.nextPoisson(1.);
    }
    return new WeightedLoss<LocalLoss>(loss, poissonWeights);
  }

  public static Computable<Vec, ? extends Func> targetByName(final String name) {
    try {
      @SuppressWarnings("unchecked")
      Class<Func> oracleClass = (Class<Func>)Class.forName("com.spbsu.ml.loss." + name);
      final Constructor<Func> constructor = oracleClass.getConstructor(Vec.class);
      return new Computable<Vec, Func>() {
        @Override
        public Func compute(Vec argument) {
          try {
            return constructor.newInstance(argument);
          } catch (Exception e) {
            throw new RuntimeException("Exception during metric " + name + " initialization", e);
          }
        }
      };
    }
    catch (Exception e) {
      throw new RuntimeException("Unable to create requested target: " + name, e);
    }
  }

  public enum NormalizationType {
    SPHERE,
    PCA,
    SCALE
  }

  public static class NormalizationProperties {
    public Vec xMean;
    public Mx xTrans;
    public double yMean;
    public double yVar;
  }

  public static DataSet normalize(DataSet ds, NormalizationType type, NormalizationProperties props) {
    final Vec mean = new ArrayVec(ds.xdim());
    final Mx covar = new VecBasedMx(ds.xdim(), ds.xdim());
    double targetMean;
    double targetVar;
    Mx trans;
    {
      DSIterator it = ds.iterator();
      double tSum = 0.;
      double tSum2 = 0.;
      while (it.advance()) {
        VecTools.append(mean, it.x());
        tSum += it.y();
        tSum2 += it.y() * it.y();
      }
      targetMean = tSum / ds.power();
      targetVar = Math.sqrt((tSum2 - ds.power() * targetMean * targetMean) / ds.power());
      VecTools.scale(mean, -1./ds.power());
    }
    Vec temp = new ArrayVec(ds.xdim());
    for (int i = 0; i < ds.power(); i++) {
      Vec vec = ds.data().row(i);
      VecTools.assign(temp, vec);
      VecTools.append(temp, mean);
      VecTools.addOuter(covar, temp, temp);
    }
    VecTools.scale(covar, 1./ds.power());
    switch (type) {
      case SPHERE:
        final Mx l = VecTools.choleskyDecomposition(covar);
        trans = VecTools.inverseLTriangle(l);
        break;
      case PCA:
        trans = new VecBasedMx(ds.xdim(), ds.xdim());
        VecTools.eigenDecomposition(covar, new VecBasedMx(ds.xdim(), ds.xdim()), trans);
        break;
      case SCALE:
        trans = new VecBasedMx(ds.xdim(), ds.xdim());
        for (int i = 0; i < trans.columns(); i++) {
          trans.set(i, i, 1./Math.sqrt(covar.get(i, i)));
        }
        break;
      default:
        throw new NotImplementedException();
    }
    Vec newTarget = VecTools.copy(ds.target());
    Mx newData = VecTools.copy(ds.data());
    for (int i = 0; i < ds.power(); i++) {
      Vec row = newData.row(i);
      VecTools.append(row, mean);
      VecTools.assign(row, VecTools.multiply(trans, row));

      newTarget.set(i, (newTarget.get(i) - targetMean) / targetVar);
    }
    props.xMean = mean;
    props.xTrans = trans;
    props.yMean = targetMean;
    props.yVar = targetVar;
    return new DataSetImpl(newData, newTarget);
  }

}
