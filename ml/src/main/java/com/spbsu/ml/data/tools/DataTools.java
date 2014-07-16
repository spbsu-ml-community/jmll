package com.spbsu.ml.data.tools;

import java.io.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.zip.GZIPInputStream;


import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.func.types.ConversionRepository;
import com.spbsu.commons.func.types.SerializationRepository;
import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.system.RuntimeUtils;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.func.TransJoin;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.JsonFeatureMeta;
import com.spbsu.ml.meta.impl.JsonPoolMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.models.ObliviousMultiClassTree;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.list.array.TIntArrayList;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:05
 */
public class DataTools {
  public static Pool<QURLItem> loadFromFeaturesTxt(String file) throws IOException {
    return loadFromFeaturesTxt(file, file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file));
  }

  public static FeaturesTxtPool loadFromFeaturesTxt(String file, Reader in) throws IOException {
    final List<QURLItem> items = new ArrayList<>();
    final VecBuilder target = new VecBuilder();
    final VecBuilder data = new VecBuilder();
    final int[] featuresCount = new int[]{-1};
    CharSeqTools.processLines(in, new Processor<CharSequence>() {
      int lindex = 0;
      @Override
      public void process(final CharSequence arg) {
        lindex++;
        final CharSequence[] parts = CharSeqTools.split(arg, '\t');
        items.add(new QURLItem(CharSeqTools.parseInt(parts[0]), parts[2].toString(), CharSeqTools.parseInt(parts[3])));
        target.append(CharSeqTools.parseDouble(parts[1]));
        if (featuresCount[0] < 0)
          featuresCount[0] = parts.length - 4;
        else if (featuresCount[0] != parts.length - 4)
          throw new RuntimeException("\"Failed to parse line \" + lindex + \":\"");
        for (int i = 4; i < parts.length; i++) {
          data.append(CharSeqTools.parseDouble(parts[i]));
        }
      }
    });
    return new FeaturesTxtPool(file,
        new ArraySeq<>(items.toArray(new QURLItem[items.size()])),
        new VecBasedMx(featuresCount[0], data.build()),
        target.build());
  }

  public static void writeModel(Computable result, File to) throws IOException {
    BFGrid grid = grid(result);
    StreamTools.writeChars(CharSeqTools.concat(result.getClass().getCanonicalName(), "\t", Boolean.toString(grid != null), "\n",
        SERIALIZATION.write(result)), to);
  }

  public static Trans readModel(String fileName, ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    final LineNumberReader modelReader = new LineNumberReader(new InputStreamReader(new FileInputStream(fileName)));
    String line = modelReader.readLine();
    CharSequence[] parts = CharSeqTools.split(line, '\t');
    Class<? extends Trans> modelClazz = (Class<? extends Trans>)Class.forName(parts[0].toString());
    return serializationRepository.read(StreamTools.readReader(modelReader), modelClazz);
  }

  public static BFGrid grid(Computable<?, Vec> result) {
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

  public static DataSet extendDataset(VecDataSet sourceDS, Mx addedColumns) {
    Vec[] columns = new Vec[addedColumns.columns()];
    for (int i = 0; i < addedColumns.columns(); i++) {
      columns[i] = addedColumns.col(i);
    }
    return extendDataset(sourceDS, columns);
  }

  public static VecDataSet extendDataset(VecDataSet sourceDS, Vec... addedColumns) {
    if (addedColumns.length == 0)
      return sourceDS;

    Mx oldData = sourceDS.data();
    Mx newData = new VecBasedMx(oldData.rows(), oldData.columns() + addedColumns.length);
    for (MxIterator iter = oldData.nonZeroes(); iter.advance(); ) {
      newData.set(iter.row(), iter.column(), iter.value());
    }
    for (int i = 0; i < addedColumns.length; i++) {
      for (VecIterator iter = addedColumns[i].nonZeroes(); iter.advance(); ) {
        newData.set(iter.index(), oldData.columns() + i, iter.value());
      }
    }
    return new VecDataSetImpl(newData, sourceDS);
  }

  public static Vec value(Mx ds, Func f) {
    Vec result = new ArrayVec(ds.rows());
    for (int i = 0; i < ds.rows(); i++) {
      result.set(i, f.value(ds.row(i)));
    }
    return result;
  }

  public static <LocalLoss extends StatBasedLoss> WeightedLoss<LocalLoss> bootstrap(LocalLoss loss, FastRandom rnd) {
    int[] poissonWeights = new int[loss.xdim()];
    for (int i = 0; i < loss.xdim(); i++) {
      poissonWeights[i] = rnd.nextPoisson(1.);
    }
    return new WeightedLoss<LocalLoss>(loss, poissonWeights);
  }

  public static Class<? extends TargetFunc> targetByName(final String name) {
    try {
      return (Class<? extends TargetFunc>)Class.forName("com.spbsu.ml.loss." + name);
    }
    catch (Exception e) {
      throw new RuntimeException("Unable to create requested target: " + name, e);
    }
  }

  public static final SerializationRepository<CharSequence> SERIALIZATION = new SerializationRepository<>(
      new TypeConvertersCollection(MathTools.CONVERSION, "com.spbsu.ml.io"), CharSequence.class);

  public static int[][] splitAtRandom(final int size, final FastRandom rng, final double... v) {
    Vec weights = new ArrayVec(v);
    TIntArrayList[] folds = new TIntArrayList[v.length];
    for (int i = 0; i < size; i++) {
      folds[rng.nextSimple(weights)].add(i);
    }
    int[][] result = new int[folds.length][];
    for (int i = 0; i < folds.length; i++) {
      result[i] = folds[i].toArray();
    }
    return result;
  }

  public static <T> Vec calcAll(final Computable<T, Vec> result, final DataSet<T> data) {
    VecBuilder results = new VecBuilder(data.length());
    int dim = 0;
    for (int i = 0; i < data.length(); i++) {
      final Vec vec = result.compute(data.at(i));
      for (int j = 0; j < vec.length(); j++) {
        results.add(vec.at(j));
      }
      dim = vec.length();
    }
    return dim > 1 ? new VecBasedMx(dim, results.build()) : results.build();
  }

  public static <Target extends TargetFunc> Target newTarget(final Class<Target> targetClass, final Seq<?> values, final DataSet<?> ds) {
    try {
      final Constructor<Target>[] constructors = (Constructor<Target>[])targetClass.getConstructors();
      for (Constructor<Target> constructor : constructors) {
        final Class<?>[] parameters = constructor.getParameterTypes();
        if (parameters.length == 2 &&
            parameters[0].isAssignableFrom(values.getClass()) &&
            parameters[1].isAssignableFrom(ds.getClass()))
          return constructor.newInstance(values, ds);
      }
      throw new RuntimeException("No propper constructor!");
    } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
      throw new RuntimeException(e);
    }
  }

  public static enum LineType {
    ITEMS,
    FEATURE,
    TARGET,
  }

  public static <T extends DSItem> void writeTo(final Pool<T> pool, Writer out) throws IOException {
    final JsonFactory jsonFactory = new JsonFactory();
    jsonFactory.disable(JsonGenerator.Feature.QUOTE_FIELD_NAMES);
    jsonFactory.configure(JsonParser.Feature.ALLOW_COMMENTS, false);
    { // meta
      out.append(LineType.ITEMS.name().toLowerCase()).append('\t');
      StringWriter writer = new StringWriter();
      final JsonGenerator generator = jsonFactory.createGenerator(writer);
      generator.writeStartObject();
      generator.writeStringField("author", pool.meta().author());
      generator.writeStringField("source", pool.meta().source());
      generator.writeNumberField("created", pool.meta().created().getTime());
      generator.writeEndObject();
      generator.close();
      out.append(writer.getBuffer());
      for (int i = 0; i < pool.size(); i++) {
        out.append('\t').append(SERIALIZATION.write(pool.items.at(i)));
      }
      out.append('\n');
    }

    for (int i = 0; i < pool.features.length; i++) { // features
      final Pair<? extends FeatureMeta, ? extends Seq<?>> feature = pool.features[i];
      out.append(LineType.FEATURE.name().toLowerCase()).append('\t');
      writeFeature(out, jsonFactory, feature);
    }

    { // target
      out.append(LineType.TARGET.name().toLowerCase()).append('\t');
      writeFeature(out, jsonFactory, pool.target);
    }
  }

  private static void writeFeature(final Writer out, final JsonFactory jsonFactory,
                                   final Pair<? extends FeatureMeta, ? extends Seq<?>> feature) throws IOException
  {
    {
      StringWriter writer = new StringWriter();
      final JsonGenerator generator = jsonFactory.createGenerator(writer);
      generator.writeStartObject();
      generator.writeStringField("id", feature.first.id());
      generator.writeStringField("description", feature.first.description());
      generator.writeStringField("type", feature.first.type().name());
      generator.writeEndObject();
      generator.close();
      out.append(writer.getBuffer());
    }
    out.append('\t');
    out.append(SERIALIZATION.write(feature.getSecond()));
    out.append('\n');
  }

  public static <T extends DSItem> Pool<T> loadFromFile(Reader input, final Class<T> dsiClass) throws IOException{
    try {
      final PoolBuilder builder = new PoolBuilder();
      CharSeqTools.processLines(input, new Processor<CharSequence>() {
        @Override
        public void process(final CharSequence arg) {
          final CharSequence[] parts = CharSeqTools.split(arg, '\t');
          try {
            switch (LineType.valueOf(parts[0].toString().toUpperCase())) {
              case ITEMS: {
                final JsonParser parser = CharSeqTools.parseJSON(parts[1]);
                builder.setMeta(parser.readValueAs(JsonPoolMeta.class));
                int index = 2;
                while (index < parts.length) {
                  builder.addItem(SERIALIZATION.read(parts[index], dsiClass));
                }
                break;
              }
              case FEATURE: {
                final JsonParser parser = CharSeqTools.parseJSON(parts[1]);
                JsonFeatureMeta fmeta = parser.readValueAs(JsonFeatureMeta.class);

                Class<? extends Seq<?>> vecClass = fmeta.type().clazz();
                builder.newFeature(fmeta, SERIALIZATION.read(
                    CharSeqTools.concatWithDelimeter("\t", Arrays.asList(parts).subList(2, parts.length)),
                    vecClass));
                break;
              }
              case TARGET: {
                final JsonParser parser = CharSeqTools.parseJSON(parts[1]);
                JsonTargetMeta fmeta = parser.readValueAs(JsonTargetMeta.class);

                Class<? extends Seq<?>> vecClass = fmeta.type().clazz();
                builder.newTarget(fmeta, SERIALIZATION.read(
                    CharSeqTools.concatWithDelimeter("\t", Arrays.asList(parts).subList(2, parts.length)),
                    vecClass));
                break;
              }
            }
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      });
      return builder.create(dsiClass);
    }
    catch (RuntimeException e) {
      if (e.getCause() instanceof IOException) {
        throw (IOException)e.getCause();
      }
      throw e;
    }
  }

  public static <T extends DSItem> Pool<T> loadFromFile(String file, Class<T> dsiClass) throws IOException {
    try(final InputStreamReader input =
        file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file)) {
      return loadFromFile(
          input, dsiClass);
    }
  }

}
