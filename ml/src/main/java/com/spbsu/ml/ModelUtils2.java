package com.spbsu.ml;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;


import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.text.CharSequenceTools;
import com.spbsu.ml.io.ModelsSerializationRepository;

/**
 * User: starlight
 * Date: 07.11.13
 */
//TODO starlight: Move to JMLL repo
public class ModelUtils2 {
  private ModelUtils2() { }

  public static Trans readModel(final InputStream inputStream, final ModelsSerializationRepository serializationRepository) throws
                                                                                                                            IOException,
                                                                                                                            ClassNotFoundException
  {
    final LineNumberReader modelReader = new LineNumberReader(new InputStreamReader(inputStream));
    final String line = modelReader.readLine();
    final CharSequence[] parts = CharSequenceTools.split(line, '\t');
    //noinspection unchecked
    final Class<? extends Trans> modelClazz = (Class<? extends Trans>) Class.forName(parts[0].toString());
    return serializationRepository.read(StreamTools.readReader(modelReader), modelClazz);
  }

  public static Trans readModel(final InputStream modelInputStream, final InputStream gridInputStream) throws IOException,
                                                                                                              ClassNotFoundException
  {
    final ModelsSerializationRepository repository = new ModelsSerializationRepository();
    final BFGrid grid = repository.read(StreamTools.readStream(gridInputStream), BFGrid.class);
    final ModelsSerializationRepository customizedRepository = repository.customizeGrid(grid);
    return readModel(modelInputStream, customizedRepository);
  }
}
