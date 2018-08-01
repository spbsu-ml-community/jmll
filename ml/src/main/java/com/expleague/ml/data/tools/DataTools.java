package com.expleague.ml.data.tools;


import com.expleague.commons.func.types.SerializationRepository;
import com.expleague.commons.func.types.impl.TypeConvertersCollection;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.mx.MxByRowsBuilder;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.IndexTransVec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.*;
import com.expleague.commons.system.RuntimeUtils;
import com.expleague.commons.text.StringUtils;
import com.expleague.commons.util.JSONTools;
import com.expleague.commons.util.logging.Logger;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.CompositeTrans;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.dynamicGrid.models.ObliviousTreeDynamicBin;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.BFGrid;
import com.expleague.ml.io.ModelsSerializationRepository;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.meta.GroupedDSItem;
import com.expleague.ml.meta.PoolFeatureMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;
import com.expleague.ml.meta.items.FakeItem;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.ml.models.ObliviousTree;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.AnnotationIntrospector;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.fasterxml.jackson.module.jaxb.JaxbAnnotationIntrospector;
import com.expleague.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.TransJoin;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.models.ObliviousMultiClassTree;
import com.expleague.ml.models.multilabel.MultiLabelBinarizedModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import org.apache.commons.lang3.mutable.MutableInt;

import java.io.*;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:05
 */
@SuppressWarnings("unused")
public class DataTools {
  public static Logger log = Logger.create(DataTools.class);

  public static final SerializationRepository<CharSequence> SERIALIZATION = new SerializationRepository<>(
      new TypeConvertersCollection(MathTools.CONVERSION, DataTools.class, "com.expleague.ml.io"), CharSequence.class);


  public static Pool<QURLItem> loadFromFeaturesTxt(final String file) throws IOException {
    return loadFromFeaturesTxt(file, file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file));
  }

  public static FeaturesTxtPool loadFromFeaturesTxt(final String fileName, final Reader in) throws IOException {
    final List<QURLItem> items = new ArrayList<>();
    final VecBuilder target = new VecBuilder();
    final VecBuilder data = new VecBuilder();
    final int[] featuresCount = new int[]{-1};
    CharSeqTools.processLines(in, new Consumer<CharSequence>() {
      int lindex = 0;

      @Override
      public void accept(final CharSequence arg) {
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
    return new FeaturesTxtPool(
        new ArraySeq<>(items.toArray(new QURLItem[items.size()])),
        new VecBasedMx(featuresCount[0], data.build()),
        target.build()
    );
  }


  public static int getLineCount(final Reader input, final char sep) {
    return CharSeqTools.lines(input, false).limit(1).map(line -> CharSeqTools.split(line, sep).length).findFirst().orElse(0);
  }

  public static double stringToDoubleHash(final CharSequence in) {
    final long hashCode = in.toString().hashCode();
    return hashCode * 1.0 / (1L << 32);
  }

  public static CatboostPool loadFromCatBoostPool(final CatBoostPoolDescription poolDescription,
                                                  final Reader in) throws IOException {
    final VecBuilder target = new VecBuilder();
    final VecBuilder data = new VecBuilder();

    CharSeqTools.processLines(in, new Consumer<CharSequence>() {
      int lindex = -1;

      @Override
      public void accept(final CharSequence arg) {
        lindex++;
        if (lindex == 0 && poolDescription.hasHeaderColumn()) {
          return;
        }

        final CharSequence[] parts = CharSeqTools.split(arg, poolDescription.getDelimiter());
        if (parts.length != poolDescription.columnCount()) {
          throw new RuntimeException("\"Failed to parse line \" + lindex + \":\"");
        }

        int id = lindex;

        for (int column = 0; column < poolDescription.columnCount(); ++column) {
          final CharSequence columnSeq = parts[column];
          switch (poolDescription.columnType(column)) {
            case Target: {
              target.append(CharSeqTools.parseDouble(columnSeq));
              break;
            }
            case Num: {
              data.append(CharSeqTools.parseDouble(columnSeq));
              break;
            }
            case Categ: {
              final double value = stringToDoubleHash(columnSeq);
              if (Double.isNaN(value)) {
                throw new RuntimeException("Error: catFeature hash values should not be NaN");
              }
              data.append(value);
              break;
            }
            case Weight: {
              throw new RuntimeException("Unimplemented yet");
            }
            case DocId:
            case Auxiliary:
            case QueryId:
            default: {
              break;
            }
          }
        }
      }
    });

    final Set<Integer> catFeatureIds = new TreeSet<>();
    int factorId = 0;
    for (int column = 0; column < poolDescription.columnCount(); ++column) {
      final CatBoostPoolDescription.ColumnType columnType = poolDescription.columnType(column);
      if (columnType == CatBoostPoolDescription.ColumnType.Categ) {
        catFeatureIds.add(factorId);
      }
      if (CatBoostPoolDescription.ColumnType.isFactorColumn(columnType)) {
        ++factorId;
      }
    }
    final Mx vecData = new VecBasedMx(poolDescription.factorCount(), data.build());
    final Vec targetVec = target.build();
    return new CatboostPool(vecData, targetVec, catFeatureIds);
  }

  public static void writeModel(final Function result, final File to) throws IOException {
    final BFGrid grid = grid(result);
    StreamTools.writeChars(CharSeqTools.concat(result.getClass().getCanonicalName(), "\t", Boolean.toString(grid != null), "\n",
        SERIALIZATION.write(result)), to);
  }

  public static <T extends Function> T readModel(final InputStream inputStream, final ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    final LineNumberReader modelReader = new LineNumberReader(new InputStreamReader(inputStream));
    final String line = modelReader.readLine();
    final CharSequence[] parts = CharSeqTools.split(line, '\t');
    //noinspection unchecked
    final Class<? extends Function> modelClazz = (Class<? extends Function>) Class.forName(parts[0].toString());
    //noinspection unchecked
    return (T) serializationRepository.read(StreamTools.readReader(modelReader), modelClazz);
  }

  public static <T extends Function> T readModel(final String fileName, final ModelsSerializationRepository serializationRepository) throws IOException, ClassNotFoundException {
    return readModel(new FileInputStream(fileName), serializationRepository);
  }

  public static <T extends Function> T readModel(final InputStream modelInputStream, final InputStream gridInputStream) throws IOException, ClassNotFoundException {
    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final BFGrid grid = repository.read(StreamTools.readStream(gridInputStream), BFGridImpl.class);
    final ModelsSerializationRepository customizedRepository = repository.customizeGrid(grid);
    return readModel(modelInputStream, customizedRepository);
  }

  public static void writeBinModel(final Function result, final File file) {
    if (result instanceof Ensemble) {
      //noinspection unchecked
      final Ensemble<Trans> ensemble = (Ensemble) result;
      if (ensemble.models.length == 0)
        return;
      if (ensemble.models[0] instanceof ObliviousTreeDynamicBin) {
        final DynamicGrid grid = dynamicGrid(ensemble);
        final DynamicBinModelBuilder builder = new DynamicBinModelBuilder(Objects.requireNonNull(grid));
        for (int i = 0; i < ensemble.models.length; ++i) {
          builder.append((ObliviousTreeDynamicBin) ensemble.models[i], ensemble.weights.at(i));
        }
        builder.build().toFile(file);
      } else if (ensemble.models[0] instanceof ObliviousTree) {
        final BFGrid grid = grid(ensemble);
        final BinModelBuilder builder = new BinModelBuilder(Objects.requireNonNull(grid));
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
    }
    catch (ClassNotFoundException e) {
      return Pair.create(false, "Invalid model : " + e.getCause());
    }
  }

  public static Pair<Boolean, String> validateModel(final String filePath, final ModelsSerializationRepository repository) throws IOException {
    return validateModel(new FileInputStream(filePath), repository);
  }

  public static DynamicGrid dynamicGrid(final Function<?,Vec> result) {
    if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      return dynamicGrid(ensemble.last());
    }
    else if (result instanceof ObliviousTreeDynamicBin) {
      return ((ObliviousTreeDynamicBin) result).grid();
    }
    return null;
  }

  public static BFGrid grid(final Function<?,Vec> result) {
    if (result instanceof CompositeTrans) {
      final CompositeTrans composite = (CompositeTrans) result;
      BFGrid grid = grid(composite.f);
      grid = grid == null ? grid(composite.g) : grid;
      return grid;
    }
    else if (result instanceof FuncJoin) {
      final FuncJoin join = (FuncJoin) result;
      for (final Func dir : join.dirs()) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    }
    else if (result instanceof TransJoin) {
      final TransJoin join = (TransJoin) result;
      for (final Trans dir : join.dirs) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    }
    else if (result instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) result;
      for (final Trans dir : ensemble.models) {
        final BFGrid grid = grid(dir);
        if (grid != null)
          return grid;
      }
    }
    else if (result instanceof MultiClassModel) {
      return grid(((MultiClassModel) result).getInternModel());
    }
    else if (result instanceof MultiLabelBinarizedModel) {
      return grid(((MultiLabelBinarizedModel) result).getInternModel());
    }
    else if (result instanceof ObliviousTree) {
      return ((ObliviousTree) result).grid();
    }
    else if (result instanceof ObliviousMultiClassTree) {
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
    return new WeightedLoss<>(loss, poissonWeights);
  }

  public static Class<? extends TargetFunc> targetByName(final String name) {
    try {
      //noinspection unchecked
      return (Class<? extends TargetFunc>) Class.forName("com.expleague.ml.loss." + name);
    }
    catch (Exception e) {
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

  public static <D extends DSItem> List<Pool<D>> splitDataSet(Pool<D> pool, final FastRandom rng, final double... v) {
    final int[][] cvSplit = DataTools.splitAtRandom(pool.size(), rng, v);
    final List<Pool<D>> result = new ArrayList<>();
    for (int i = 0; i < v.length; i++) {
      result.add(pool.sub(cvSplit[i]));
    }
    return result;
  }

  public static <D extends GroupedDSItem> List<Pool<D>> splitGroupDataSet(Pool<D> pool, final FastRandom rng, final double... v) {
    final TObjectIntMap<String> groups = new TObjectIntHashMap<>();
    final TIntObjectMap<String> rgroups = new TIntObjectHashMap<>();
    for (int i = 0; i < pool.size(); i++) {
      final String group = pool.items.at(i).groupId();
      if (!groups.containsKey(group)) {
        rgroups.put(groups.size(), group);
        groups.put(group, groups.size());
      }
    }

    final int[][] cvSplit = DataTools.splitAtRandom(groups.size(), rng, v);
    final List<Pool<D>> result = new ArrayList<>();
    for (int i = 0; i < v.length; i++) {
      final TIntList indices = new TIntArrayList();
      final Set<String> currentGroups = Arrays.stream(cvSplit[i]).mapToObj(rgroups::get).collect(Collectors.toSet());

      for (int j = 0; j < pool.size(); j++) {
        final String group = pool.items.at(j).groupId();
        if (currentGroups.contains(group))
          indices.add(j);
      }

      result.add(pool.sub(indices.toArray()));
    }
    return result;
  }

  public static <T> Vec calcAll(final Function<T, Vec> result, final DataSet<T> data) {
    final VecBuilder results = new VecBuilder(data.length());
    int dim = 0;
    for (int i = 0; i < data.length(); i++) {
      final Vec vec = result.apply(data.at(i));
      for (int j = 0; j < vec.length(); j++) {
        results.add(vec.at(j));
      }
      dim = vec.length();
    }
    return dim > 1 ? new VecBasedMx(dim, results.build()) : results.build();
  }

  public static <Target extends TargetFunc> Target newTarget(final Class<Target> targetClass, final Seq<?> values, final DataSet<?> ds) {
    Target target;
    target = RuntimeUtils.newInstanceByAssignable(targetClass, values, ds);
    if (target != null)
      return target;
    throw new RuntimeException("No proper constructor!");
  }

  public static <T extends DSItem> void writePoolTo(final Pool<T> pool, final Writer out) throws IOException {
    final JsonFactory jsonFactory = new JsonFactory();
    jsonFactory.disable(JsonGenerator.Feature.QUOTE_FIELD_NAMES);
    jsonFactory.configure(JsonParser.Feature.ALLOW_COMMENTS, false);
    final JsonGenerator generator = jsonFactory.createGenerator(out);
    { // meta
      out.append("items").append('\t');
      final ObjectMapper mapper = new ObjectMapper(jsonFactory);
      final AnnotationIntrospector introspector =
          new JaxbAnnotationIntrospector(mapper.getTypeFactory());
      {
        mapper.setAnnotationIntrospector(introspector);
        generator.writeStartObject();
        generator.writeStringField("id", pool.meta().id());
        generator.writeStringField("author", pool.meta().author());
        generator.writeStringField("source", pool.meta().source());
        generator.writeNumberField("created", pool.meta().created().getTime());
        generator.writeStringField("type", pool.meta().type().getCanonicalName());
        generator.writeEndObject();
        generator.flush();
      }

      out.append('\t');
      {
        generator.setCodec(mapper);
        generator.writeStartArray();

        for (int i = 0; i < pool.size(); i++) {
          generator.writeObject(pool.data().at(i));
        }
        generator.writeEndArray();
        generator.flush();
      }
      out.append('\n');
    }

    for (int i = 0; i < pool.fcount(); i++) { // features
      out.append("feature").append('\t');
      writeFeature(out, jsonFactory, pool.fmeta(i), pool.fdata(i));
    }

    for (int i = 0; i < pool.tcount(); i++) { // targets
      out.append("target").append('\t');
      writeFeature(out, jsonFactory, pool.tmeta(i), pool.tdata(i));
    }
    out.flush();
    generator.close();
  }

  private static void writeFeature(final Writer out, final JsonFactory jsonFactory,
                                   final PoolFeatureMeta meta, Seq<?> values) throws IOException {
    {
      final StringWriter writer = new StringWriter();
      final JsonGenerator generator = jsonFactory.createGenerator(writer);
      generator.writeStartObject();
      generator.writeStringField("id", meta.id());
      generator.writeStringField("description", meta.description());
      generator.writeStringField("type", meta.type().name());
      generator.writeStringField("associated", meta.associated().meta().id());
      generator.writeEndObject();
      generator.close();
      out.append(writer.getBuffer());
    }
    out.append('\t');
    out.append(SERIALIZATION.write(values));
    out.append('\n');
  }


  public static <T extends DSItem> Pool<T> readPoolFrom(final Reader input) throws IOException {
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
            final CollectionType itemsGroupType = mapper.getTypeFactory().constructCollectionType(List.class, meta.type());
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
      //noinspection unchecked
      return (Pool<T>) builder.create();
    }
    catch (RuntimeException e) {
      if (e.getCause() instanceof IOException) {
        throw (IOException) e.getCause();
      }
      throw e;
    }
  }

  public static Pool<? extends DSItem> loadFromFile(final String fileName) throws IOException {
    return loadFromFile(new File(fileName));
  }

  public static Reader gzipOrFileReader(final File file) throws IOException {
    return file.getName().endsWith(".gz") ?
        new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) :
        new FileReader(file);
  }

  public static Pool<? extends DSItem> loadFromFile(final File file) throws IOException {
    try (final Reader input = gzipOrFileReader(file)) {
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
    for (int i = 0; i < pool.fcount(); i++) {
      builder
          .append("\n")
          .append("feature #").append(i)
          .append(": type = ").append(pool.fmeta(i).type());
    }
    return builder.toString();
  }

  public static Pool<FakeItem> loadFromLibSvmFormat(final Reader in) throws IOException {
    final MutableInt poolFeaturesCount = new MutableInt(-1);

    final VecBuilder targetBuilder = new VecBuilder();
    final List<Pair<TIntList, TDoubleList>> features = new ArrayList<>();

    CharSeqTools.processLines(in, new Consumer<CharSequence>() {
      int lindex = 0;

      @Override
      public void accept(final CharSequence arg) {
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

    return FakePool.create(dataMx, targetBuilder.build());
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
