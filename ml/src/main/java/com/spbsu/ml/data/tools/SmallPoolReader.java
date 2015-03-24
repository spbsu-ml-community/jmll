package com.spbsu.ml.data.tools;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.CollectionType;
import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.Processor;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.util.JSONTools;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;
import com.spbsu.ml.meta.impl.JsonFeatureMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;

import java.io.*;
import java.util.List;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

/**
 * Created by irlab on 27.01.2015.
 */
public final class SmallPoolReader {
  private SmallPoolReader() {
  }

  public static final TypeConverter<String, SparseVec> string2SparseVecConverter = new String2SparseVecConverter();
  public static final TypeConverter<String, Vec> string2VecConverter = new String2VecConverter();

  public static class String2SparseVecConverter implements TypeConverter<String, SparseVec> {
    @Override
    public SparseVec convert(final String from) {
      final StringTokenizer tokenizer = new StringTokenizer(from, " ");

      String token = tokenizer.nextToken();
      int pos = token.indexOf(':');
      String index = token.substring(0, pos);
      String value = token.substring(pos + 1);

      final int dim = Integer.parseInt(index);
      final int nzCount = Integer.parseInt(value);
      final int[] indices = new int[nzCount];
      final double[] values = new double[nzCount];
      for (int i = 0; tokenizer.hasMoreTokens(); i++) {
        token = tokenizer.nextToken();
        pos = token.indexOf(':');
        index = token.substring(0, pos);
        value = token.substring(pos + 1);
        indices[i] = Integer.parseInt(index);
        values[i] = Double.parseDouble(value);
      }
      return new SparseVec(dim, indices, values);
    }
  }

  public static class String2VecConverter implements TypeConverter<String, Vec> {
    @Override
    public Vec convert(final String from) {
      final StringTokenizer tokenizer = new StringTokenizer(from, " ");
      final int size = Integer.parseInt(tokenizer.nextToken());
      final Vec result = new ArrayVec(size);
      for (int i = 0; tokenizer.hasMoreTokens(); i++) {
        final String token = tokenizer.nextToken();
        result.set(i, Double.parseDouble(token));
      }
      return result;
    }
  }

  public static Pool<? extends DSItem> readPoolFrom(final Reader input) throws IOException {
    return readPoolFrom(input, null);
  }

  public static Pool<? extends DSItem> readPoolFrom(final Reader input, final Filter<JsonFeatureMeta> featureFilter) throws IOException {
    try {
      final PoolBuilder builder = new PoolBuilder();
      final Processor<String[]> seqProcessor = new Processor<String[]>() {
        @Override
        public void process(final String[] parts) {
          try {
            final JsonParser parser = JSONTools.parseJSON(parts[1]);
            switch (parts[0]) {
              case "items": {
                final JsonDataSetMeta meta = parser.readValueAs(JsonDataSetMeta.class);
                builder.setMeta(meta);

                final JsonParser parseItems = JSONTools.parseJSON(parts[2]);
                final ObjectMapper mapper = (ObjectMapper) parseItems.getCodec();
                final CollectionType itemsGroupType = mapper.getTypeFactory().constructCollectionType(List.class, meta.type().clazz());
                final List<? extends DSItem> myObjects = mapper.readValue(parseItems, itemsGroupType);
                for (final DSItem myObject : myObjects) {
                  builder.addItem(myObject);
                }
                break;
              }
              case "feature": {
                final JsonFeatureMeta fmeta = parser.readValueAs(JsonFeatureMeta.class);
                if (featureFilter != null && !featureFilter.accept(fmeta))
                  break;
                final TypeConverter<String, ? extends Vec> typeConverter = fmeta.type() == FeatureMeta.ValueType.SPARSE_VEC ? string2SparseVecConverter : string2VecConverter;
                builder.newFeature(fmeta, typeConverter.convert(parts[2]));
                break;
              }
              case "target": {
                final JsonTargetMeta fmeta = parser.readValueAs(JsonTargetMeta.class);
                final TypeConverter<String, ? extends Vec> typeConverter = fmeta.type() == FeatureMeta.ValueType.SPARSE_VEC ? string2SparseVecConverter : string2VecConverter;
                builder.newTarget(fmeta, typeConverter.convert(parts[2]));
                break;
              }
            }
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      };
      final LineNumberReader lineNumberReader = new LineNumberReader(input);
      for (String line = lineNumberReader.readLine(); line != null; line = lineNumberReader.readLine()) {
        final String[] split = line.split("\t", 3);
        seqProcessor.process(split);
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
    return loadFromFile(new File(fileName), null);
  }

  public static Pool<? extends DSItem> loadFromFile(final String fileName, final Filter<JsonFeatureMeta> featureFilter) throws IOException {
    return loadFromFile(new File(fileName), featureFilter);
  }

  public static Pool<? extends DSItem> loadFromFile(final File file) throws IOException {
    return loadFromFile(file, null);
  }

  public static Pool<? extends DSItem> loadFromFile(final File file, final Filter<JsonFeatureMeta> featureFilter) throws IOException {
    try (final InputStreamReader input = file.getName().endsWith(".gz") ?
            new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) :
            new FileReader(file)) {
      return readPoolFrom(input, featureFilter);
    }
  }
}
