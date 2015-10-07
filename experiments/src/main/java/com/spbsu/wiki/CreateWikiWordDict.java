package com.spbsu.wiki;

import com.spbsu.commons.io.Buffer;
import com.spbsu.commons.io.codec.IntExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.IntSeq;
import it.unimi.dsi.fastutil.bytes.ByteArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
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
            //args = [cts, path input file, path output file, path to indexes file]
            final FileWriter fw = new FileWriter(args[2]);
            BufferedReader reader = new BufferedReader(new FileReader(args[1]));
            String line;
            while((line = reader.readLine()) != null){
                IntSeq seq = WikiUtils.stringToIntSeq(line.toLowerCase());
                StringBuilder sb = new StringBuilder();
                for(int i : seq.arr){
                    sb.append(i).append(" ");
                }
                fw.write(sb.toString().trim() + "\n");
            }
            WikiUtils.writeIndexesInFile(new File(args[3]));
            fw.close();
        }
        if(args[0].equals("cd")) { //coding
            //args = [cd, path input file, path output file]
            final IntExpansion coding = new IntExpansion(500000);
            WikiUtils.setDefaultCoding(coding);
            BufferedReader bf = new BufferedReader(new FileReader(args[1]));
            String line;
            String path = args[3].substring(0, args[3].lastIndexOf("."));
            int iter = 0;
            while((line = bf.readLine()) != null){
                System.out.print("\r" + iter);
                IntArrayList doc = new IntArrayList();
                for(String i : line.split("\\s"))
                    doc.add(Integer.parseInt(i));
                coding.accept(new IntSeq(doc.toIntArray()));
                if(++iter % 1000000 == 0){
                    Dictionary<Integer> dictionary = WikiUtils.getCurrentDictionary();
                    FileWriter fw = new FileWriter(path + "_" + iter + ".dict");
                    for (int j = 0; j < dictionary.alphabet().size(); j++) {
                        Object sequence = dictionary.get(j);
                        String str = "";
                        IntSeq seq = (IntSeq)sequence;
                        if(seq.length() > 1) {
                            for (int i = 0; i < seq.length(); i++)
                                str += seq.at(i) + " ";
                            fw.write(str.trim() + "\t" + coding.expansion().resultFreqs()[j] + "\n");
                        }
                    }
                    fw.close();
                }
            }
        }
        if(args[0].equals("ctw")) { //convert dictionary to words
            //args = [cd, path input file, path output file, path to indexes file]
            WikiUtils.readIndexes(new File(args[3]));
            BufferedReader bf = new BufferedReader(new FileReader(args[1]));
            String line;
            FileWriter fw = new FileWriter(args[2]);
            while((line = bf.readLine()) != null){
                IntArrayList doc = new IntArrayList();
                for(String i : line.split("\t")[0].split("\\s"))
                    doc.add(Integer.parseInt(i));
                String text = WikiUtils.intSeqToWords(doc.toIntArray());
                fw.write(text + "\t" + line.split("\t")[1] + "\n");
            }
            fw.close();
        }
    }
}
