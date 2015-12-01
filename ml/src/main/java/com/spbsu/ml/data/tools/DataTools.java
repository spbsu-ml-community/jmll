package com.spbsu.ml.data.tools;


import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.func.types.SerializationRepository;
import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.math.vectors.impl.mx.MxByRowsBuilder;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.ReaderChopper;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.system.RuntimeUtils;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.JSONTools;
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
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.PoolFeatureMeta;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;
import com.spbsu.ml.meta.impl.JsonFeatureMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;
import com.spbsu.ml.meta.items.FakeItem;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.models.ObliviousMultiClassTree;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.multilabel.MultiLabelBinarizedModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.list.linked.TIntLinkedList;
import org.apache.commons.lang3.mutable.MutableInt;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.zip.GZIPInputStream;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:05
 */
public class DataTools {
  public static final SerializationRepository<CharSequence> SERIALIZATION = new SerializationRepository<>(
      new TypeConvertersCollection(MathTools.CONVERSION, "com.spbsu.ml.io"), CharSequence.class);


  public static Pool<QURLItem> loadFromFeaturesTxt(final String file) throws IOException {
    return loadFromFeaturesTxt(file, file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file));
  }

  public static FeaturesTxtPool loadFromFeaturesTxt(final String fileName, final Reader in) throws IOException {
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
    return new FeaturesTxtPool(fileName,
        new ArraySeq<>(items.toArray(new QURLItem[items.size()])),
        new VecBasedMx(featuresCount[0], data.build()),
        target.build());
  }

  public static void writeModel(final Computable result, final File to) throws IOException {
    final BFGrid grid = grid(result);
    StreamTools.writeChars(CharSeqTools.concat(result.getClass().getCanonicalName(), "\t", Boolean.toString(grid != null), "\n",
        SERIALIZATION.write(result)), to);
  }

  public static Trans readModel(final InputStream inputStream, final ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    final LineNumberReader modelReader = new LineNumberReader(new InputStreamReader(inputStream));
    final String line = modelReader.readLine();
    final CharSequence[] parts = CharSeqTools.split(line, '\t');
    //noinspection unchecked
    final Class<? extends Trans> modelClazz = (Class<? extends Trans>) Class.forName(parts[0].toString());
    return serializationRepository.read(StreamTools.readReader(modelReader), modelClazz);
  }

  public static Trans readModel(final String fileName, final ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    return readModel(new FileInputStream(fileName), serializationRepository);
  }

  public static Trans readModel(final InputStream modelInputStream, final InputStream gridInputStream) throws IOException, ClassNotFoundException {
    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final BFGrid grid = repository.read(StreamTools.readStream(gridInputStream), BFGrid.class);
    final ModelsSerializationRepository customizedRepository = repository.customizeGrid(grid);
    return readModel(modelInputStream, customizedRepository);
  }

  public static void writeBinModel(final Computable result, final File file) throws IOException {
    if (result instanceof Ensemble) {
      final Ensemble<Trans> ensemble = (Ensemble) result;
      if (ensemble.models.length == 0)
        return;
      if (ensemble.models[0] instanceof ObliviousTreeDynamicBin) {
        final DynamicGrid grid = dynamicGrid(ensemble);
        final DynamicBinModelBuilder builder = new DynamicBinModelBuilder(grid);
        for (int i = 0; i < ensemble.models.length; ++i) {
          builder.append((ObliviousTreeDynamicBin) ensemble.models[i], ensemble.weights.at(i));
        }
        builder.build().toFile(file);
      } else if (ensemble.models[0] instanceof ObliviousTree) {
        final BFGrid grid = grid(ensemble);
        final BinModelBuilder builder = new BinModelBuilder(grid);
        for (int i = 0; i < ensemble.models.length; ++i) {
          builder.append((ObliviousTree) ensemble.models[i], ensemble.weights.at(i));
        }
        builder.build().toFile(file);
      }
    }
  }

  public static Pair<Boolean, String> validateModel(final InputStream modelInputStream, final ModelsSerializationRepository repository) throws IOException {
    try {
      final Trans trans = readModel(modelInputStream, repository);
      return Pair.create(true, "Valid model : " + trans.getClass().getSimpleName());
    } catch (ClassNotFoundException e) {
      return Pair.create(false, "Invalid model : " + e.getCause());
    }
  }

  public static Pair<Boolean, String> validateModel(final String filePath, final ModelsSerializationRepository repository) throws IOException {
    return validateModel(new FileInputStream(filePath), repository);
  }

  public static DynamicGrid dynamicGrid(final Computable<?, Vec> result) {
    if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      return dynamicGrid(ensemble.last());
    } else if (result instanceof ObliviousTreeDynamicBin) {
      return ((ObliviousTreeDynamicBin) result).grid();
    }
    return null;
  }
  public static BFGrid grid(final Computable<?, Vec> result) {
    if (result instanceof CompositeTrans) {
      final CompositeTrans composite = (CompositeTrans) result;
      BFGrid grid = grid(composite.f);
      grid = grid == null ? grid(composite.g) : grid;
      return grid;
    } else if (result instanceof FuncJoin) {
      final FuncJoin join = (FuncJoin) result;
      for (final Func dir : join.dirs()) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    } else if (result instanceof TransJoin) {
      final TransJoin join = (TransJoin) result;
      for (final Trans dir : join.dirs) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    } else if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      for (final Trans dir : ensemble.models) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    } else if (result instanceof MultiClassModel) {
      return grid(((MultiClassModel) result).getInternModel());
    } else if (result instanceof MultiLabelBinarizedModel) {
      return grid(((MultiLabelBinarizedModel) result).getInternModel());
    } else if (result instanceof ObliviousTree) {
      return ((ObliviousTree)result).grid();
    } else if (result instanceof ObliviousMultiClassTree) {
      return ((ObliviousMultiClassTree) result).binaryClassifier().grid();
    }
    return null;
  }

  public static DataSet extendDataset(final VecDataSet sourceDS, final Mx addedColumns) {
    final Vec[] columns = new Vec[addedColumns.columns()];
    for (int i = 0; i < addedColumns.columns(); i++) {
      columns[i] = addedColumns.col(i);
    }
    return extendDataset(sourceDS, columns);
  }

  public static VecDataSet extendDataset(final VecDataSet sourceDS, final Vec... addedColumns) {
    if (addedColumns.length == 0)
      return sourceDS;

    final Mx oldData = sourceDS.data();
    final Mx newData = new VecBasedMx(oldData.rows(), oldData.columns() + addedColumns.length);
    for (final MxIterator iter = oldData.nonZeroes(); iter.advance(); ) {
      newData.set(iter.row(), iter.column(), iter.value());
    }
    for (int i = 0; i < addedColumns.length; i++) {
      for (final VecIterator iter = addedColumns[i].nonZeroes(); iter.advance(); ) {
        newData.set(iter.index(), oldData.columns() + i, iter.value());
      }
    }
    return new VecDataSetImpl(newData, sourceDS);
  }

  public static Vec value(final Mx ds, final Func f) {
    final Vec result = new ArrayVec(ds.rows());
    for (int i = 0; i < ds.rows(); i++) {
      result.set(i, f.value(ds.row(i)));
    }
    return result;
  }

  public static <LocalLoss extends StatBasedLoss> WeightedLoss<LocalLoss> bootstrap(final LocalLoss loss, final FastRandom rnd) {
    final int[] poissonWeights = new int[loss.xdim()];
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

  public static int[][] splitAtRandom(final int size, final FastRandom rng, final double... v) {
    final Vec weights = new ArrayVec(v);
    final TIntList[] folds = new TIntList[v.length];
    for (int i = 0; i < folds.length; i++) {
      folds[i] = new TIntLinkedList();
    }
    for (int i = 0; i < size; i++) {
      folds[rng.nextSimple(weights)].add(i);
    }
    final int[][] result = new int[folds.length][];
    for (int i = 0; i < folds.length; i++) {
      result[i] = folds[i].toArray();
    }
    return result;
  }

  public static <T> Vec calcAll(final Computable<T, Vec> result, final DataSet<T> data) {
    final VecBuilder results = new VecBuilder(data.length());
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

  public static <T extends DSItem> void writePoolTo(final Pool<T> pool, final Writer out) throws IOException {
    final JsonFactory jsonFactory = new JsonFactory();
    jsonFactory.disable(JsonGenerator.Feature.QUOTE_FIELD_NAMES);
    jsonFactory.configure(JsonParser.Feature.ALLOW_COMMENTS, false);
    { // meta
      out.append("items").append('\t');
      {
        final StringWriter writer = new StringWriter();
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
        final StringWriter writer = new StringWriter();
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
    out.flush();
  }

  private static void writeFeature(final Writer out, final JsonFactory jsonFactory,
                                   final Pair<? extends PoolFeatureMeta, ? extends Seq<?>> feature) throws IOException {
    {
      final StringWriter writer = new StringWriter();
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


  public static Pool<? extends DSItem> readPoolFrom(final Reader input) throws IOException {
    try {
      final PoolBuilder builder = new PoolBuilder();
      final ReaderChopper chopper = new ReaderChopper(input);
      CharSequence name;
      while ((name = chopper.chop('\t')) != null) {
        if (name.length() == 0)
          continue;
        final JsonParser parser = JSONTools.parseJSON(chopper.chop('\t'));

        switch (name.toString()) {
          case "items": {
            final JsonDataSetMeta meta = parser.readValueAs(JsonDataSetMeta.class);
            builder.setMeta(meta);

            final JsonParser parseItems = JSONTools.parseJSON(chopper.chop('\n'));
            final ObjectMapper mapper = (ObjectMapper) parseItems.getCodec();
            mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
            final CollectionType itemsGroupType = mapper.getTypeFactory().constructCollectionType(List.class, meta.type().clazz());
            final List<? extends DSItem> myObjects = mapper.readValue(parseItems, itemsGroupType);
            for (int i = 0; i < myObjects.size(); i++) {
              builder.addItem(myObjects.get(i));
            }
            break;
          }
          case "feature": {
            final JsonFeatureMeta fmeta = parser.readValueAs(JsonFeatureMeta.class);
            final Class<? extends Seq<?>> vecClass = fmeta.type().clazz();
            builder.newFeature(fmeta, SERIALIZATION.read(
                    chopper.chop('\n'),
                    vecClass));
            break;
          }
          case "target": {
            final JsonTargetMeta fmeta = parser.readValueAs(JsonTargetMeta.class);
            final Class<? extends Seq<?>> vecClass = fmeta.type().clazz();
            builder.newTarget(fmeta, SERIALIZATION.read(
                    chopper.chop('\n'),
                    vecClass));
            break;
          }
        }
      }
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

  public static <S extends Seq<?>> Pair<VecDataSet, S> createSubset(final VecDataSet sourceDS, final S sourceTarget, final int[] idxs) {
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

  public static String getPoolInfo(final Pool<?> pool) {
    final VecDataSet vecDataSet = pool.vecData();

    final StringBuilder builder = new StringBuilder()
        .append("Pool size = ").append(pool.size())
        .append("\n")
        .append("VecDS features count = ").append(vecDataSet.xdim())
        .append("\n");
    for (int i = 0; i < pool.features.length; i++) {
      builder
          .append("\n")
          .append("feature #").append(i)
          .append(": type = ").append(pool.features[i].getFirst().type());
    }
    return builder.toString();
  }

  public static Pool<FakeItem> loadFromLibSvmFormat(final Reader in) throws IOException {
    final MutableInt poolFeaturesCount = new MutableInt(-1);

    final VecBuilder targetBuilder = new VecBuilder();
    final List<Pair<TIntList, TDoubleList>> features = new ArrayList<>();

    CharSeqTools.processLines(in, new Processor<CharSequence>() {
      int lindex = 0;

      @Override
      public void process(final CharSequence arg) {
        final CharSequence[] parts = CharSeqTools.split(arg, ' ');

        targetBuilder.add(CharSeqTools.parseDouble(parts[0]));

        final TIntList rowIndices = new TIntLinkedList();
        final TDoubleList rowValues = new TDoubleLinkedList();

        for (int i = 1; i < parts.length; i++) {
          final CharSequence indexAndValue = parts[i];
          if (StringUtils.isBlank(indexAndValue)) {
            continue;
          }

          final CharSequence[] split = CharSeqTools.split(indexAndValue, ':');
          final int index = CharSeqTools.parseInt(split[0]);
          final double value = CharSeqTools.parseDouble(split[1]);
          rowIndices.add(index);
          rowValues.add(value);

          if (poolFeaturesCount.intValue() < index + 1) {
            poolFeaturesCount.setValue(index + 1);
          }
        }
        features.add(Pair.create(rowIndices, rowValues));
        lindex++;
      }
    });

    final MxBuilder mxBuilder = new MxByRowsBuilder();
    for (Pair<TIntList, TDoubleList> pair : features) {
      mxBuilder.add(new SparseVec(poolFeaturesCount.intValue(), pair.getFirst().toArray(), pair.getSecond().toArray()));
    }
    final Mx dataMx = mxBuilder.build();

    return new FakePool(dataMx, targetBuilder.build());
  }

  public static void writePoolInLibfmFormat(final Pool<?> pool, final Writer out) throws IOException {
    final Mx data = pool.vecData().data();
    final Vec target = pool.target(L2.class).target;
    for (int i = 0; i < pool.size(); i++) {
      final double t = target.get(i);
      out.append(String.valueOf(t));
      final VecIterator vecIterator = data.row(i).nonZeroes();
      while (vecIterator.advance()) {
        out.append("\t")
           .append(String.valueOf(vecIterator.index()))
           .append(":")
           .append(String.valueOf(vecIterator.value()));
      }
      out.append("\n");
    }
    out.flush();
  }

  public static void writeClassicPoolTo(final Pool<?> pool, final String fileName) throws IOException {
    DataTools.writeClassicPoolTo(pool, new BufferedWriter(new FileWriter(fileName)));
  }

  public static void writeClassicPoolTo(final Pool<?> pool, final Writer out) throws IOException {
    final DecimalFormat preciseFormatter = new DecimalFormat("###.########", new DecimalFormatSymbols(Locale.US));

    final Mx vecData = pool.vecData().data();
    final Vec target = pool.target(L2.class).target;

    for (int i = 0; i < vecData.rows(); i++) {
      out.write(String.format("%d\t%s\turl\t0", i, preciseFormatter.format(target.get(i))));
      for (int j = 0; j < vecData.columns(); j++) {
        out.append("\t").append(preciseFormatter.format(vecData.get(i, j)));
      }
      out.write("\n");
    }
    out.flush();
  }
}
