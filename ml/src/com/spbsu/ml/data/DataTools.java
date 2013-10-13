package com.spbsu.ml.data;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.text.CharSequenceTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.models.AdditiveModel;
import com.spbsu.ml.models.AdditiveMultiClassModel;
import com.spbsu.ml.models.ObliviousMultiClassTree;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.TIntArrayList;
import gnu.trove.TIntObjectHashMap;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
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
    while ((line = reader.readLine()) != null) {
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

  public static ChangedTarget changeTarget(DataSet base, Vec newTarget) {
    return new ChangedTarget((DataSetImpl)base, newTarget);
  }

  public static Bootstrap bootstrap(DataSet base) {
    return new Bootstrap(base);
  }

  public static Bootstrap bootstrap(DataSet base, Random random) {
    return new Bootstrap(base, random);
  }

  public static void writeModel(Model result, File to, ModelsSerializationRepository serializationRepository) throws IOException {
    BFGrid grid = grid(result);
    StreamTools.writeChars(CharSequenceTools.concat(result.getClass().getCanonicalName(), "\t", Boolean.toString(grid != null), "\n",
                           serializationRepository.write(result)), to);
  }

  public static Model readModel(String fileName, ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    final LineNumberReader modelReader = new LineNumberReader(new InputStreamReader(new FileInputStream(fileName)));
    String line = modelReader.readLine();
    CharSequence[] parts = CharSequenceTools.split(line, '\t');
    Class<? extends Model> modelClazz = (Class<? extends Model>)Class.forName(parts[0].toString());
    return serializationRepository.read(StreamTools.readReader(modelReader), modelClazz);
  }

  public static BFGrid grid(Model result) {
    if (result instanceof AdditiveModel)
      return grid((Model)((AdditiveModel) result).models.get(0));
    if (result instanceof ObliviousTree)
      return ((ObliviousTree)result).grid();
    if (result instanceof AdditiveMultiClassModel)
      return grid((Model)((AdditiveMultiClassModel) result).models.get(0));
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

  public static int[] extractClass(Vec target, int classNo) {
    final TIntArrayList points = new TIntArrayList(target.dim());
    for (int i = 0; i < target.dim(); i++) {
      if (target.get(i) == classNo)
        points.add(i);
    }
    return points.toNativeArray();
  }

  public static DataSet normalizeClasses(DataSet learn) {
    final DSIterator it = learn.iterator();
    final Vec normalized = new ArrayVec(learn.power());
    while (it.advance()) {
      normalized.set(it.index(), normalizeRelevance(it.y()));
    }
    return new ChangedTarget((DataSetImpl)learn, normalized);
  }

  private static double normalizeRelevance(double y) {
    if (y < 0.07)
      return 0.;
//    else if (y < 0.14)
//      return 1.;
//    else if (y < 0.41)
//      return 2.;
//    else if (y < 0.61)
//      return 3.;
//    else
//      return 4.;
    return 1.;
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
