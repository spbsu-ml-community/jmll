package com.spbsu.wiki;

import com.spbsu.commons.io.codec.IntExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.seq.ArraySeq;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;

import java.io.*;
import java.util.Map;

/**
 * Created by Юлиан on 05.10.2015.
 */
public class CreateWikiWordDict {

    public static void main(String[] args) throws Exception {
        if(args[0].equals("p")) { //parse dump
            final FileWriter[] fw = new FileWriter[1];
            fw[0] = new FileWriter(args[2]);
            final int[] counter = new int[]{0};
            final WikiParser parser = new WikiforiaParser();
            parser.setProcessor((String str) -> {
                try {
                    System.out.print("\r" + counter[0]++);
                    if(str.length() > 100)
                        fw[0].write(str + "\n");
                    if (counter[0] % 100000 == 0) {
                        fw[0].close();
                        fw[0] = new FileWriter(args[2], true);
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                }

            });
            parser.parse(new BZip2CompressorInputStream(
                    new FileInputStream(args[1])));
        }
        if(args[0].equals("s")) { //split into sentences
            final FileWriter fw = new FileWriter(args[2]);
            BufferedReader reader = new BufferedReader(new FileReader(args[1]));
            String line;
            int n = 0;
            while((line = reader.readLine()) != null){
                if(n++ % 1000 == 0)
                    System.out.print("\r" + n);
                for(String s : WikiUtils.splitIntoSentences(line)){
                    fw.write(s + "\n");
                }
            }
            fw.close();
        }
        if(args[0].equals("rp")) { //remove punctuations
            final FileWriter fw = new FileWriter(args[2]);
            BufferedReader reader = new BufferedReader(new FileReader(args[1]));
            String line;
            while((line = reader.readLine()) != null){
                fw.write(WikiUtils.removePunctuation(line) + "\n");
            }
            fw.close();
        }
        if(args[0].equals("cts")) { //convert phrases to IntSeq
            int[] seq = new int[]{1,2,3};
            int[] seq2 = new int[]{4,5,6};
            PrintWriter pw = new PrintWriter("A:\\test.bin");

            pw.print(WikiUtils.intArrayToByteArray(seq));
            pw.print(WikiUtils.intArrayToByteArray(seq));
            pw.close();
        }
        if(args[0].equals("cd")) { //coding
            final IntExpansion coding = new IntExpansion(500000);
            WikiUtils.setDefaultCoding(coding);
        }
    }
}
