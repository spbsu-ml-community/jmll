package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.math.vectors.impl.vectors.IndexTransVec;
import com.spbsu.commons.util.ArrayTools;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TIntLinkedList;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;


import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.func.types.SerializationRepository;
import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxIterator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
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
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.func.TransJoin;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.PoolFeatureMeta;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;
import com.spbsu.ml.meta.impl.JsonFeatureMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.models.ObliviousMultiClassTree;
import com.spbsu.ml.models.ObliviousTree;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TIntLinkedList;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

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

  public static void writeBinModel(Computable result, File file) throws IOException {
    if (result instanceof Ensemble) {
      Ensemble<Trans> ensemble = (Ensemble) result;
      if (ensemble.models.length == 0)
        return;
      if (ensemble.models[0] instanceof ObliviousTreeDynamicBin) {
        DynamicGrid grid = dynamicGrid(ensemble);
        DynamicBinModelBuilder builder = new DynamicBinModelBuilder(grid);
        for (int i = 0; i < ensemble.models.length; ++i) {
          builder.append((ObliviousTreeDynamicBin) ensemble.models[i], ensemble.weights.at(i));
        }
        builder.build().toFile(file);
      } else if (ensemble.models[0] instanceof ObliviousTree) {
        BFGrid grid = grid(ensemble);
        BinModelBuilder builder = new BinModelBuilder(grid);
        for (int i = 0; i < ensemble.models.length; ++i) {
          builder.append((ObliviousTree) ensemble.models[i], ensemble.weights.at(i));
        }
        builder.build().toFile(file);
      }
    }
  }


  public static DynamicGrid dynamicGrid(Computable<?, Vec> result) {
    if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      return dynamicGrid(ensemble.last());
    } else if (result instanceof ObliviousTreeDynamicBin) {
      return ((ObliviousTreeDynamicBin) result).grid();
    }
    return null;
  }
  public static BFGrid grid(Computable<?, Vec> result) {
    if (result instanceof CompositeTrans) {
      final CompositeTrans composite = (CompositeTrans) result;
      BFGrid grid = grid(composite.f);
      grid = grid == null ? grid(composite.g) : grid;
      return grid;
    } else if (result instanceof FuncJoin) {
      final FuncJoin join = (FuncJoin) result;
      for (Func dir : join.dirs()) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    } else if (result instanceof TransJoin) {
      final TransJoin join = (TransJoin) result;
      for (Trans dir : join.dirs) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    } else if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      for (Trans dir : ensemble.models) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    } else if (result instanceof ObliviousTree)
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
      return (Class<? extends TargetFunc>) Class.forName("com.spbsu.ml.loss." + name);
    } catch (Exception e) {
      throw new RuntimeException("Unable to create requested target: " + name, e);
    }
  }

  public static final SerializationRepository<CharSequence> SERIALIZATION = new SerializationRepository<>(
      new TypeConvertersCollection(MathTools.CONVERSION, "com.spbsu.ml.io"), CharSequence.class);

  public static int[][] splitAtRandom(final int size, final FastRandom rng, final double... v) {
    final Vec weights = new ArrayVec(v);
    final TIntList[] folds = new TIntList[v.length];
    for (int i = 0; i < folds.length; i++) {
      folds[i] = new TIntLinkedList();
    }
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

  @Nullable
  public static <Target extends TargetFunc> Target newTarget(final Class<Target> targetClass, final Seq<?> values, final DataSet<?> ds) {
    Target target = null;
    target = RuntimeUtils.newInstanceByAssignable(targetClass, values, ds);
    if (target != null)
      return target;
    throw new RuntimeException("No proper constructor!");
  }

  public static <T extends DSItem> void writePoolTo(final Pool<T> pool, Writer out) throws IOException {
    final JsonFactory jsonFactory = new JsonFactory();
    jsonFactory.disable(JsonGenerator.Feature.QUOTE_FIELD_NAMES);
    jsonFactory.configure(JsonParser.Feature.ALLOW_COMMENTS, false);
    { // meta
      out.append("items").append('\t');
      {
        StringWriter writer = new StringWriter();
        final JsonGenerator generator = jsonFactory.createGenerator(writer);
        generator.writeStartObject();
        generator.writeStringField("id", pool.meta().id());
        generator.writeStringField("author", pool.meta().author());
        generator.writeStringField("source", pool.meta().source());
        generator.writeNumberField("created", pool.meta().created().getTime());
        generator.writeStringField("type", pool.meta().type().name());
        generator.writeEndObject();
        generator.close();
        out.append(writer.getBuffer());
      }

      out.append('\t');
      {
        StringWriter writer = new StringWriter();
        final JsonGenerator generator = jsonFactory.createGenerator(writer);
        generator.setCodec(new ObjectMapper(jsonFactory));
        generator.writeStartArray();

        for (int i = 0; i < pool.size(); i++) {
          generator.writeObject(pool.data().at(i));
        }
        generator.writeEndArray();
        generator.close();
        out.append(writer.getBuffer());
      }
      out.append('\n');
    }

    for (int i = 0; i < pool.features.length; i++) { // features
      out.append("feature").append('\t');
      writeFeature(out, jsonFactory, pool.features[i]);
    }

    for (int i = 0; i < pool.targets.size(); i++) { // targets
      out.append("target").append('\t');
      writeFeature(out, jsonFactory, pool.targets.get(i));
    }
  }

  private static void writeFeature(final Writer out, final JsonFactory jsonFactory,
                                   final Pair<? extends PoolFeatureMeta, ? extends Seq<?>> feature) throws IOException {
    {
      StringWriter writer = new StringWriter();
      final JsonGenerator generator = jsonFactory.createGenerator(writer);
      generator.writeStartObject();
      generator.writeStringField("id", feature.first.id());
      generator.writeStringField("description", feature.first.description());
      generator.writeStringField("type", feature.first.type().name());
      generator.writeStringField("associated", feature.first.associated().meta().id());
      generator.writeEndObject();
      generator.close();
      out.append(writer.getBuffer());
    }
    out.append('\t');
    out.append(SERIALIZATION.write(feature.getSecond()));
    out.append('\n');
  }

  public static Pool<? extends DSItem> readPoolFrom(Reader input) throws IOException {
    try {
      final PoolBuilder builder = new PoolBuilder();
      CharSeqTools.processAndSplitLines(input, new Processor<CharSequence[]>() {
            @Override
            public void process(final CharSequence[] parts) {
              try {
                final JsonParser parser = CharSeqTools.parseJSON(parts[1]);
                switch (parts[0].toString()) {
                  case "items": {
                    final JsonDataSetMeta meta = parser.readValueAs(JsonDataSetMeta.class);
                    builder.setMeta(meta);

                    final JsonParser parseItems = CharSeqTools.parseJSON(parts[2]);
                    final ObjectMapper mapper = (ObjectMapper) parseItems.getCodec();
                    final CollectionType itemsGroupType = mapper.getTypeFactory().constructCollectionType(List.class, meta.type().clazz());
                    final List<? extends DSItem> myObjects = mapper.readValue(parseItems, itemsGroupType);
                    for (int i = 0; i < myObjects.size(); i++) {
                      builder.addItem(myObjects.get(i));
                    }
                    break;
                  }
                  case "feature": {
                    JsonFeatureMeta fmeta = parser.readValueAs(JsonFeatureMeta.class);
                    Class<? extends Seq<?>> vecClass = fmeta.type().clazz();
                    builder.newFeature(fmeta, SERIALIZATION.read(
                        CharSeqTools.concatWithDelimeter("\t", Arrays.asList(parts).subList(2, parts.length)),
                        vecClass));
                    break;
                  }
                  case "target": {
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
          },
          "\t", true);
      return builder.create();
    } catch (RuntimeException e) {
      if (e.getCause() instanceof IOException) {
        throw (IOException) e.getCause();
      }
      throw e;
    }
  }

  public static Pool<? extends DSItem> loadFromFile(final String fileName) throws IOException {
    return loadFromFile(new File(fileName));
  }

  public static Pool<? extends DSItem> loadFromFile(final File file) throws IOException {
    try (final InputStreamReader input = file.getName().endsWith(".gz") ?
        new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) :
        new FileReader(file)) {
      return readPoolFrom(input);
    }
  }

  public static <S extends Seq<?>> Pair<VecDataSet, S> createSubset(final VecDataSet sourceDS, final S sourceTarget, int[] idxs) {
    final VecDataSet subSet = new VecDataSetImpl(
        new VecBasedMx(
            sourceDS.xdim(),
            new IndexTransVec(sourceDS.data(),
                new RowsPermutation(
                    idxs,
                    sourceDS.xdim()
                )
            )
        ),
        sourceDS
    );
    final S subTarget = (S) ArrayTools.cut((Seq<?>) sourceTarget, idxs);
    return Pair.create(subSet, subTarget);
  }
}
