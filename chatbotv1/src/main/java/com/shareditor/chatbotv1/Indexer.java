package com.shareditor.chatbotv1;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.wltea.analyzer.lucene.IKAnalyzer;

public class Indexer 
{
    public static void main( String[] args ) throws IOException
    {
        if (args.length != 2) {
        	System.err.println("Usage: " + Indexer.class.getSimpleName() + " corpus_path index_path");
        	System.exit(-1);
        }
        
        String corpusPath = args[0];
        String indexPath = args[1];
        
        Analyzer analyzer = new IKAnalyzer(true);
        IndexWriterConfig iwc = new IndexWriterConfig(Version.LUCENE_4_9, analyzer);
		iwc.setOpenMode(OpenMode.CREATE);
		iwc.setUseCompoundFile(true);
		IndexWriter indexWriter = new IndexWriter(FSDirectory.open(new File(indexPath)), iwc);
		
		BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(corpusPath), "UTF-8"));
        String line = "";
        String last = "";
        long lineNum = 0;
        while ((line = br.readLine()) != null) {
        	line = line.trim();
        	
        	if (0 == line.length()) {
        		continue;
        	}
        	
        	if (!last.equals("")) {
        		Document doc = new Document();
        		doc.add(new TextField("question", last, Store.YES));
        		doc.add(new StoredField("answer", line));
        		indexWriter.addDocument(doc);
        	}
        	last = line;
        	lineNum++;
        	if (lineNum % 100000 == 0) {
        		System.out.println("add doc " + lineNum);
        	}
        }
		br.close();
		
		indexWriter.forceMerge(1);
		indexWriter.close();
    }
}
