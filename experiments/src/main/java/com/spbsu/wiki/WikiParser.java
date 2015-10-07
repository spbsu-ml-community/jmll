package com.spbsu.wiki;

import com.spbsu.commons.func.Processor;
import com.spbsu.commons.seq.IntSeq;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * Created by Юлиан on 05.10.2015.
 */
public interface WikiParser {

    void parse(final InputStream stream) throws Exception;
    void setProcessor(final Processor<String> str);

}
